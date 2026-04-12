import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
import numpy as np

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms
from dann_modules import DomainDiscriminator

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DANN (Domain Adversarial) Training on {device}")

    # 1. Datasets
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Models
    # Using ResNet-101 Unet
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    ).to(device)

    # Discriminator: ResNet-101 has 2048 channels at the bottleneck
    # We will use Global Average Pooling on the encoder features
    discr = DomainDiscriminator(input_dim=2048).to(device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_path}")

    # 3. Losses & Optimizers
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    tversky_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
    lovasz_loss = smp.losses.LovaszLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def seg_criterion(preds, targets):
        return 0.5 * tversky_loss(preds, targets) + 0.5 * lovasz_loss(preds, targets) + ce_loss(preds, targets)

    bce_loss = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(list(model.parameters()) + list(discr.parameters()), lr=args.lr)
    
    # 4. Training Loop
    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    num_steps = args.epochs * len(src_loader)
    current_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        discr.train()
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc="DANN-Train")
        for batch in pbar:
            # Update Alpha for GRL: starts at 0 and grows to 1
            p = float(current_step) / num_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            current_step += 1

            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            trg_imgs = trg_batch['image'].to(device)

            optimizer.zero_grad()

            # --- 1. Source Domain ---
            # Forward through encoder to get bottleneck features
            # In smp.Unet, model.encoder(x) returns a list of features
            src_features_list = model.encoder(src_imgs)
            src_bottleneck = src_features_list[-1] # [B, 2048, H/32, W/32]
            src_pooled = torch.mean(src_bottleneck, dim=(2, 3)) # Global Average Pooling
            
            # Segmentation Head
            src_outputs = model.decoder(src_features_list)
            src_outputs = model.segmentation_head(src_outputs)
            loss_seg = seg_criterion(src_outputs, src_masks)
            
            # Domain loss (Source = 0)
            src_domain_preds = discr(src_pooled, alpha=alpha)
            loss_domain_src = bce_loss(src_domain_preds, torch.zeros_like(src_domain_preds))

            # --- 2. Target Domain ---
            trg_features_list = model.encoder(trg_imgs)
            trg_bottleneck = trg_features_list[-1]
            trg_pooled = torch.mean(trg_bottleneck, dim=(2, 3))
            
            # Domain loss (Target = 1)
            trg_domain_preds = discr(trg_pooled, alpha=alpha)
            loss_domain_trg = bce_loss(trg_domain_preds, torch.ones_like(trg_domain_preds))

            # Total Loss
            total_loss = loss_seg + args.lambda_domain * (loss_domain_src + loss_domain_trg)
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({'seg_l': f"{loss_seg.item():.3f}", 'dom_l': f"{(loss_domain_src+loss_domain_trg).item():.3f}", 'alpha': f"{alpha:.2f}"})
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_dann.pth")
            print(f"Saved Best DANN Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda_domain", type=float, default=0.1) # Weight for domain loss
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
