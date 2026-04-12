import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms
from ddsp import DistributionDisruptionModule, disruption_consistency_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DDSP (Distribution Disruption) Training on {device}")

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

    # DDSP Module
    ddm = DistributionDisruptionModule(p=args.disruption_prob).to(device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_path}")

    # 3. Losses & Optimizer
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    tversky_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
    lovasz_loss = smp.losses.LovaszLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def seg_criterion(preds, targets):
        return 0.5 * tversky_loss(preds, targets) + 0.5 * lovasz_loss(preds, targets) + ce_loss(preds, targets)

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
        
        pbar = tqdm(src_loader, desc="DDSP-Train")
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

            # --- Forward Pass with Disruption ---
            # 1. Get Shallow Features (from the first block of the encoder)
            src_feats_list = model.encoder(src_imgs)
            trg_feats_list = model.encoder(trg_imgs)
            
            # Disrupt the second level features (enough to be "style" but not purely low-level)
            src_shallow = src_feats_list[1]
            trg_shallow = trg_feats_list[1]
            
            src_disrupted, trg_disrupted = ddm(src_shallow, trg_shallow)
            
            # Replace shallow features in the list
            src_feats_list_dis = list(src_feats_list)
            src_feats_list_dis[1] = src_disrupted
            
            # --- Losses ---
            # A. Segmentation Loss on Original Source
            src_outputs = model.decoder(src_feats_list)
            src_outputs = model.segmentation_head(src_outputs)
            loss_seg = seg_criterion(src_outputs, src_masks)
            
            # B. Consistency Loss on Disrupted Source
            src_outputs_dis = model.decoder(src_feats_list_dis)
            src_outputs_dis = model.segmentation_head(src_outputs_dis)
            loss_cons = disruption_consistency_loss(src_outputs, src_outputs_dis)
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_ddsp.pth")
            print(f"Saved Best DDSP Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--disruption_prob", type=float, default=0.5)
    parser.add_argument("--lambda_cons", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
