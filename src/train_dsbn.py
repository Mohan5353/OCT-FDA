import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms
from dsbn import convert_dsbn, set_model_domain

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DSBN (Domain-Specific Batch Normalization) Training on {device}")

    # 1. Datasets and Loaders
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model setup
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    ).to(device)

    # Convert all BN layers to DSBN (2 domains: 0=Cirrus, 1=Spectralis)
    convert_dsbn(model, num_domains=2)
    model.to(device)
    print("Model successfully converted to DSBN.")

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        # We need to be careful with state_dict loading for DSBN
        # but our convert_dsbn helper initializes weights from standard BN.
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        # If it's a standard model, we can use load_state_dict with strict=False 
        # but better to load manually.
        # Actually, if we just want to initialize backbone, strict=False is fine.
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_path}")

    # 3. Optimizers & Losses
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    tversky_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
    lovasz_loss = smp.losses.LovaszLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def seg_criterion(preds, targets):
        return 0.5 * tversky_loss(preds, targets) + 0.5 * lovasz_loss(preds, targets) + ce_loss(preds, targets)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 4. Labels for DSBN (0=Cirrus, 1=Spectralis)
    CIRRUS_IDX = 0
    SPECTRALIS_IDX = 1

    # 5. Training Loop
    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc="DSBN-Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            trg_imgs = trg_batch['image'].to(device)

            # --- 1. Train on Source (Cirrus) ---
            set_model_domain(model, CIRRUS_IDX)
            optimizer.zero_grad()
            src_outputs = model(src_imgs)
            loss_seg = seg_criterion(src_outputs, src_masks)
            loss_seg.backward()
            
            # --- 2. Adapt on Target (Spectralis) ---
            # We don't have masks, so we only run forward to update BN statistics
            # and potentially minimize entropy for sharpness.
            set_model_domain(model, SPECTRALIS_IDX)
            trg_outputs = model(trg_imgs)
            
            # (Optional) Minimizing entropy for target to make it "sharp"
            trg_probs = torch.softmax(trg_outputs, dim=1)
            loss_entropy = -torch.mean(torch.sum(trg_probs * torch.log(trg_probs + 1e-10), dim=1))
            
            loss_total = 0.01 * loss_entropy # Entropy loss weighting
            loss_total.backward()
            
            optimizer.step()

            pbar.set_postfix({'seg_l': f"{loss_seg.item():.4f}", 'ent_l': f"{loss_entropy.item():.4f}"})
            
        # Validation on Target
        model.eval()
        set_model_domain(model, SPECTRALIS_IDX)
        val_dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val-Spectralis"):
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device).long()
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                val_dice_metric.update(preds, masks)
        
        avg_val_dice = val_dice_metric.compute().item()
        print(f"Avg Target Dice: {avg_val_dice:.4f}")
        
        scheduler.step(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            # Save the whole model (including DSBN states)
            torch.save(model.state_dict(), "checkpoints/best_model_dsbn.pth")
            print(f"Saved Best DSBN Model with Target Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
