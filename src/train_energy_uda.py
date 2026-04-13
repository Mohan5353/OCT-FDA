import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms

def energy_score(logits, T=1.0):
    """
    Computes energy score from logits.
    Lower energy = more in-distribution / confident.
    Higher energy = out-of-distribution / uncertain.
    Energy = -T * log(sum(exp(logits / T)))
    """
    return -T * torch.logsumexp(logits / T, dim=1)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Energy-Regularized UDA Training on {device}")

    # 1. Base Model & Target Data for Pseudo-labeling
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=4
    ).to(device)

    model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    print(f"Loaded Base Model from {args.pretrained_path}")

    # 2. Datasets
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    
    # We load target data without masks for pseudo-labeling, but we need a dataset that returns masks for training later
    # For simplicity, we generate pseudo-masks on the fly during training, or pre-compute them.
    # Let's do it on the fly with a custom dataloader.
    trg_ds_unlabeled = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. Optimizer & Losses
    # Use ignore_index=255 for uncertain pixels
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    # Energy thresholding parameters (can be dynamic or fixed)
    energy_threshold = args.energy_threshold

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc="Energy-UDA-Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            trg_imgs = trg_batch['image'].to(device)

            optimizer.zero_grad()

            # --- Source Loss ---
            src_outputs = model(src_imgs)
            loss_src = ce_loss_fn(src_outputs, src_masks)

            # --- Target Energy Pseudo-labeling ---
            with torch.no_grad():
                # We should evaluate target to get pseudo-labels. In practice, model weights are updated, 
                # but we use current model to generate labels.
                trg_outputs_eval = model(trg_imgs)
                trg_energy = energy_score(trg_outputs_eval, T=args.energy_T) # [B, H, W]
                trg_preds = torch.argmax(trg_outputs_eval, dim=1) # [B, H, W]
                
                # Mask out pixels with HIGH energy (uncertain)
                # We keep pixels where energy <= energy_threshold
                pseudo_mask = torch.where(trg_energy <= energy_threshold, trg_preds, torch.tensor(255).to(device))
                
            # Compute loss on target using pseudo-labels
            trg_outputs = model(trg_imgs)
            loss_trg = ce_loss_fn(trg_outputs, pseudo_mask)

            # --- Energy Regularization Loss (Minimize energy on target) ---
            # To push target features to be in-distribution
            energy_reg = torch.mean(torch.relu(energy_score(trg_outputs, T=args.energy_T) - args.energy_margin))
            
            total_loss = loss_src + args.lambda_trg * loss_trg + args.lambda_reg * energy_reg
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'src_l': f"{loss_src.item():.3f}", 'trg_l': f"{loss_trg.item():.3f}", 'e_reg': f"{energy_reg.item():.3f}"})
            
        scheduler.step()

        # Validation
        model.eval()
        val_dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device).long()
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                val_dice_metric.update(preds, masks)
        
        avg_val_dice = val_dice_metric.compute().item()
        print(f"Epoch {epoch+1} Val Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model_energy_uda.pth")
            print(f"Saved Best Energy UDA Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--energy_T", type=float, default=1.0)
    parser.add_argument("--energy_threshold", type=float, default=-2.0) # Varies based on T and logits
    parser.add_argument("--energy_margin", type=float, default=-5.0)
    parser.add_argument("--lambda_trg", type=float, default=0.5)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
