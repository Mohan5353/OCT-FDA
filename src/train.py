import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms
from fda import FDA_source_to_target

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Datasets and Loaders
    # Source: Cirrus (Labeled) - Split into Train and Test
    source_train_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='train', split_ratio=0.8)
    source_loader = DataLoader(source_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.use_fda:
        # Target: Spectralis (Unlabeled for training FDA)
        target_train_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
        target_loader = DataLoader(target_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        target_loader = None

    # Validation: Depends on mode
    if args.use_fda:
        # If FDA, validate on all Spectralis
        val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    else:
        # If Baseline, validate on the test split of Cirrus
        val_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_val_transforms(), load_mask=True, split='test', split_ratio=0.8)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model, Criterion, Optimizer
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    ).to(device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        print(f"Loaded pretrained weights from {args.pretrained_path}")

    # Combined Loss: Tversky + Lovasz + Weighted CE
    # Focus heavily on PED (index 3)
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    tversky_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
    lovasz_loss = smp.losses.LovaszLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def criterion(preds, targets):
        return 0.5 * tversky_loss(preds, targets) + 0.5 * lovasz_loss(preds, targets) + ce_loss(preds, targets)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Scheduler: Reduce LR on plateau for fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 3. Training Loop
    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Warmup for first 3 epochs (simple linear)
        if epoch < 3:
            lr = args.lr * (epoch + 1) / 3
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warmup Learning Rate: {lr:.6f}")

        # Training
        model.train()
        train_loss = 0
        target_iter = iter(target_loader) if args.use_fda else None
        
        pbar = tqdm(source_loader, desc="Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            if args.use_fda:
                try:
                    trg_batch = next(target_iter)
                except (StopIteration, TypeError):
                    target_iter = iter(target_loader)
                    trg_batch = next(target_iter)
                
                trg_imgs = trg_batch['image'].to(device)
                
                # Match batch sizes if target batch is smaller
                if trg_imgs.shape[0] < src_imgs.shape[0]:
                    src_imgs_to_adapt = src_imgs[:trg_imgs.shape[0]]
                    src_imgs_adapted = FDA_source_to_target(src_imgs_to_adapt, trg_imgs, L=args.fda_L)
                    src_imgs = torch.cat([src_imgs_adapted, src_imgs[trg_imgs.shape[0]:]], dim=0)
                else:
                    src_imgs = FDA_source_to_target(src_imgs, trg_imgs[:src_imgs.shape[0]], L=args.fda_L)
            
            optimizer.zero_grad()
            outputs = model(src_imgs)
            loss = criterion(outputs, src_masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(source_loader)

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
        print(f"Train Loss: {avg_train_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        # Step scheduler based on Val Dice
        if epoch >= 3: # Only step after warmup
            scheduler.step(avg_val_dice)

        # Save Best Model based on Dice
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            suffix = "_fda" if args.use_fda else "_baseline"
            torch.save(model.state_dict(), f"checkpoints/best_model{suffix}.pth")
            print(f"Saved Best Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_fda", action="store_true", help="Enable Fourier Domain Adaptation")
    parser.add_argument("--fda_L", type=float, default=0.05, help="FDA mutation ratio")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained .pth weights")
    
    args = parser.parse_args()
    main(args)
