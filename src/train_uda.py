import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms

class PseudoLabeledDataset(RETOUCHDataset):
    """
    Subclass to load masks from the filtered pseudo_masks directory.
    """
    def __init__(self, data_root, vendor, transforms=None, split='all'):
        super().__init__(data_root, vendor=vendor, transforms=transforms, load_mask=True, split=split)
        # Override mask paths to point to filtered pseudo_masks
        self.mask_paths = [p.replace("cropped_masks", "pseudo_masks_filtered") for p in self.mask_paths]
        # Re-verify existence
        valid_indices = [i for i, mp in enumerate(self.mask_paths) if os.path.exists(mp)]
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.mask_paths = [self.mask_paths[i] for i in valid_indices]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"UDA Training on {device}")

    # 1. Datasets and Loaders
    # Source (Labeled)
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    
    # Target (Pseudo-labeled)
    trg_pseudo_ds = PseudoLabeledDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), split='all')
    
    # Combine
    combined_ds = ConcatDataset([src_ds, trg_pseudo_ds])
    train_loader = DataLoader(combined_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Combined Dataset Size: {len(combined_ds)} (Cirrus: {len(src_ds)}, Spectralis-Pseudo: {len(trg_pseudo_ds)})")

    # Validation (Target Ground Truth)
    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model, Criterion, Optimizer
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=4
    ).to(device)

    if args.pretrained_path:
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        print(f"Initialized with pretrained weights from {args.pretrained_path}")

    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    tversky_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7, ignore_index=255)
    lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', ignore_index=255)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    def criterion(preds, targets):
        return 0.5 * tversky_loss(preds, targets) + 0.5 * lovasz_loss(preds, targets) + ce_loss(preds, targets)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 3. Training Loop
    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc="UDA-Train")
        for batch in pbar:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        model.eval()
        val_dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val-Target"):
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device).long()
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                val_dice_metric.update(preds, masks)
        
        avg_val_dice = val_dice_metric.compute().item()
        print(f"Avg Train Loss: {train_loss/len(train_loader):.4f}, Target Dice: {avg_val_dice:.4f}")
        
        scheduler.step(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model_uda_pseudo.pth")
            print(f"Saved Best UDA Model with Target Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5) # Very low LR for fine-tuning
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
