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

def info_nce_loss(features, labels, temperature=0.07):
    """
    Class-aware InfoNCE Loss.
    features: [N, D] where N is number of sampled class centroids
    labels: [N] class labels
    """
    # Calculate cosine similarity matrix [N, N]
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Mask for positive pairs (same class)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    
    # Remove self-similarity from denominator
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(features.shape[0]).view(-1, 1).to(features.device),
        0
    )
    mask = mask * logits_mask
    
    # Numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    
    # Mean log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    return -mean_log_prob_pos.mean()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CLUDA (Contrastive Alignment) Training on {device}")

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

    # 3. Optimizers
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
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc="CLUDA-Train")
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

            # --- Forward Pass ---
            # Get features before the final head
            src_feats_enc = model.encoder(src_imgs)
            src_feats_dec = model.decoder(src_feats_enc)
            src_outputs = model.segmentation_head(src_feats_dec)
            
            trg_feats_enc = model.encoder(trg_imgs)
            trg_feats_dec = model.decoder(trg_feats_enc)
            trg_outputs = model.segmentation_head(trg_feats_dec)

            # 1. Standard Segmentation Loss (Source)
            loss_seg = seg_loss_fn(src_outputs, src_masks)

            # 2. Contrastive Loss (Alignment)
            # Pool features per class for Source
            src_centroids = []
            src_labels = []
            for c in range(4):
                mask = (src_masks == c).unsqueeze(1).float()
                if mask.sum() > 0:
                    # Global average pooling of features within the mask
                    feat_c = (src_feats_dec * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
                    src_centroids.append(feat_c)
                    src_labels.append(torch.full((src_imgs.shape[0],), c, device=device))
            
            # Pool features per class for Target (using model's own predictions as pseudo-labels)
            trg_masks = torch.argmax(trg_outputs.detach(), dim=1)
            trg_centroids = []
            trg_labels = []
            for c in range(4):
                mask = (trg_masks == c).unsqueeze(1).float()
                if mask.sum() > 0:
                    feat_c = (trg_feats_dec * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
                    trg_centroids.append(feat_c)
                    trg_labels.append(torch.full((trg_imgs.shape[0],), c, device=device))

            if len(src_centroids) > 0 and len(trg_centroids) > 0:
                all_centroids = torch.cat(src_centroids + trg_centroids, dim=0) # [N, 16]
                all_labels = torch.cat(src_labels + trg_labels, dim=0)
                loss_cont = info_nce_loss(all_centroids, all_labels)
            else:
                loss_cont = torch.tensor(0.0).to(device)

            total_loss = loss_seg + args.lambda_cont * loss_cont
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({'seg_l': f"{loss_seg.item():.3f}", 'cont_l': f"{loss_cont.item():.3f}"})
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_cluda.pth")
            print(f"Saved Best CLUDA Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda_cont", type=float, default=0.1) # Contrastive loss weight
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
