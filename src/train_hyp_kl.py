import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms
from fda import FDA_source_to_target
from hyperbolic import HyperbolicCrossEntropyLoss, hyperbolic_radius_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hyperbolic + KL DA Training on {device}")

    # 1. Datasets
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=args.embedding_dim
    ).to(device)

    if args.pretrained_path:
        state_dict = torch.load(args.pretrained_path, map_location=device)
        # Handle dict if saved as {model: ..., hyp: ...}
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Strip head if necessary
        if state_dict['segmentation_head.0.weight'].shape[0] != args.embedding_dim:
            state_dict = {k: v for k, v in state_dict.items() if 'segmentation_head' not in k}
            model.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained backbone.")
        else:
            model.load_state_dict(state_dict)
            print("Loaded full model.")

    # 3. Losses
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    hyp_criterion = HyperbolicCrossEntropyLoss(num_classes=4, embedding_dim=args.embedding_dim, weight=class_weights)
    hyp_criterion.to(device)
    
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

    optimizer = optim.Adam(list(model.parameters()) + list(hyp_criterion.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 4. Training Loop
    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc="Hyp-KL-Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            
            trg_imgs = trg_batch['image'].to(device)
            
            # --- FDA & Data Prep ---
            # 1. Stylized Source (FDA)
            if trg_imgs.shape[0] < src_imgs.shape[0]:
                src_imgs_adapted = FDA_source_to_target(src_imgs[:trg_imgs.shape[0]], trg_imgs, L=args.fda_L)
                src_imgs_fda = torch.cat([src_imgs_adapted, src_imgs[trg_imgs.shape[0]:]], dim=0)
            else:
                src_imgs_fda = FDA_source_to_target(src_imgs, trg_imgs[:src_imgs.shape[0]], L=args.fda_L)

            # --- Forward & Loss ---
            optimizer.zero_grad()
            
            # 1. Source Loss (Hyperbolic CE on FDA-stylized source)
            src_feats = model(src_imgs_fda)
            loss_src = hyp_criterion(src_feats, src_masks)
            
            # 2. Target Consistency Loss (KL Divergence)
            # Compare model predictions on raw target vs stylized target
            # Use FDA to randomize target style with another target style (augmentation)
            with torch.no_grad():
                # Mix target batch with shifted version of itself for style randomization
                trg_imgs_rand = FDA_source_to_target(trg_imgs, trg_imgs.roll(1, 0), L=args.fda_L)
                trg_probs_orig = hyp_criterion.get_probs(model(trg_imgs))
            
            trg_probs_aug = hyp_criterion.get_probs(model(trg_imgs_rand))
            
            # KL(Original || Augmented)
            # torch.nn.functional.kl_div(log_input, target)
            loss_kl = kl_loss_fn(torch.log(trg_probs_aug + 1e-10), trg_probs_orig)
            
            # 3. Hyperbolic Radius Loss (Confidence)
            loss_rad = hyperbolic_radius_loss(trg_feats if 'trg_feats' in locals() else model(trg_imgs))
            
            # Total Loss
            total_loss = loss_src + 0.1 * loss_kl + 0.05 * loss_rad
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'kl': f"{loss_kl.item():.4f}"})
            
        # Validation
        model.eval()
        val_dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device).long()
                feats = model(imgs)
                probs = hyp_criterion.get_probs(feats)
                preds = torch.argmax(probs, dim=1)
                val_dice_metric.update(preds, masks)
        
        avg_val_dice = val_dice_metric.compute().item()
        print(f"Train Loss: {train_loss/len(src_loader):.4f}, Val Dice: {avg_val_dice:.4f}")
        
        scheduler.step(avg_val_dice)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyp_state_dict': hyp_criterion.state_dict()
            }, "checkpoints/best_model_hyp_kl.pth")
            print(f"Saved Best Hyp-KL Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fda_L", type=float, default=0.05)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    args = parser.parse_args()
    main(args)
