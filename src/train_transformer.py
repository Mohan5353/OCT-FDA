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
    print(f"SegFormer (Transformer) Training on {device}")

    # 1. Datasets
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model: SegFormer with mit_b3 encoder
    model = smp.Segformer(
        encoder_name="mit_b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    ).to(device)

    # 3. Losses & Optimizer
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    tversky_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
    lovasz_loss = smp.losses.LovaszLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def criterion(preds, targets):
        return 0.5 * tversky_loss(preds, targets) + 0.5 * lovasz_loss(preds, targets) + ce_loss(preds, targets)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr) # AdamW is preferred for Transformers
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop
    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc="Transformer-Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            trg_imgs = trg_batch['image'].to(device)

            # FDA style transfer
            if trg_imgs.shape[0] < src_imgs.shape[0]:
                src_imgs_adapted = FDA_source_to_target(src_imgs[:trg_imgs.shape[0]], trg_imgs, L=args.fda_L)
                src_imgs_fda = torch.cat([src_imgs_adapted, src_imgs[trg_imgs.shape[0]:]], dim=0)
            else:
                src_imgs_fda = FDA_source_to_target(src_imgs, trg_imgs[:src_imgs.shape[0]], L=args.fda_L)

            optimizer.zero_grad()
            outputs = model(src_imgs_fda)
            
            loss = criterion(outputs, src_masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_transformer.pth")
            print(f"Saved Best Transformer Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4) # Smaller batch for transformer memory
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fda_L", type=float, default=0.05)
    
    args = parser.parse_args()
    main(args)
