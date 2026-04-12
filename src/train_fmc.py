import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
import random

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms
from fda import Fourier_Mixup

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fourier Mixup Consistency (FMC) Training on {device}")

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
        classes=4
    ).to(device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_path}")

    # 3. Losses
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    seg_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
        
        pbar = tqdm(src_loader, desc="FMC-Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            trg_imgs = trg_batch['image'].to(device)

            # Ensure trg_imgs matches src_imgs batch size
            if trg_imgs.shape[0] < src_imgs.shape[0]:
                trg_imgs = trg_imgs.repeat(2, 1, 1, 1)[:src_imgs.shape[0]]
            else:
                trg_imgs = trg_imgs[:src_imgs.shape[0]]

            # --- Fourier Mixup Consistency ---
            # Generate two differently styled versions of the same source images
            lam1 = random.uniform(0.0, args.fda_L_max)
            lam2 = random.uniform(0.0, args.fda_L_max)
            
            src_mix1 = Fourier_Mixup(src_imgs, trg_imgs, L=args.fda_window, lam=lam1)
            src_mix2 = Fourier_Mixup(src_imgs, trg_imgs, L=args.fda_window, lam=lam2)

            optimizer.zero_grad()

            # 1. Segmentation Loss on one style
            out1 = model(src_mix1)
            loss_seg = seg_loss_fn(out1, src_masks)
            
            # 2. Consistency Loss between two styles
            out2 = model(src_mix2)
            # Minimize KL divergence between probability distributions of style 1 and style 2
            prob1 = F.softmax(out1, dim=1)
            prob2 = F.softmax(out2, dim=1)
            loss_cons = F.kl_div(prob1.log(), prob2, reduction='batchmean')

            total_loss = loss_seg + args.lambda_cons * loss_cons
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'seg_l': f"{loss_seg.item():.3f}", 'cons_l': f"{loss_cons.item():.4f}"})
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_fmc.pth")
            print(f"Saved Best FMC Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fda_window", type=float, default=0.05)
    parser.add_argument("--fda_L_max", type=float, default=0.7) # Max style mix ratio
    parser.add_argument("--lambda_cons", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
