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
from hyperbolic import HyperbolicCrossEntropyLoss, hyperbolic_radius_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hyperbolic DA Training on {device}")

    # 1. Datasets and Loaders
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model
    # Important: To use Hyperbolic loss, we need features before the final classification layer.
    # smp.Unet classification head is identity by default if we ask for same classes, 
    # but we'll use it as a feature extractor.
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=args.embedding_dim # Output embeddings instead of final classes
    ).to(device)

    if args.pretrained_path:
        # Note: If pretrained has 4 classes, we only load the encoder/decoder
        state_dict = torch.load(args.pretrained_path, map_location=device)
        # Filter out the classification head weights if embedding_dim != 4
        if state_dict['segmentation_head.0.weight'].shape[0] != args.embedding_dim:
            state_dict = {k: v for k, v in state_dict.items() if 'segmentation_head' not in k}
            model.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained backbone (skipping head).")
        else:
            model.load_state_dict(state_dict)
            print("Loaded full pretrained model.")

    # 3. Hyperbolic Loss
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    hyp_criterion = HyperbolicCrossEntropyLoss(num_classes=4, embedding_dim=args.embedding_dim, weight=class_weights)
    hyp_criterion.to(device)

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
        
        pbar = tqdm(src_loader, desc="Hyper-Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            
            trg_imgs = trg_batch['image'].to(device)
            
            # Apply FDA
            if trg_imgs.shape[0] < src_imgs.shape[0]:
                src_imgs_adapted = FDA_source_to_target(src_imgs[:trg_imgs.shape[0]], trg_imgs, L=args.fda_L)
                src_imgs = torch.cat([src_imgs_adapted, src_imgs[trg_imgs.shape[0]:]], dim=0)
            else:
                src_imgs = FDA_source_to_target(src_imgs, trg_imgs[:src_imgs.shape[0]], L=args.fda_L)

            optimizer.zero_grad()
            
            # 1. Source Loss (Hyperbolic CE)
            src_feats = model(src_imgs)
            loss_src = hyp_criterion(src_feats, src_masks)
            
            # 2. Target Loss (Hyperbolic Radius Maximization)
            trg_feats = model(trg_imgs)
            loss_trg = hyperbolic_radius_loss(trg_feats)
            
            total_loss = loss_src + 0.1 * loss_trg
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device).long()
                
                feats = model(imgs)
                # Map distances back to classes for evaluation
                x_flat = feats.permute(0, 2, 3, 1).reshape(-1, args.embedding_dim)
                x_hyp = hyp_criterion.manifold.expmap0(x_flat)
                proto_hyp = hyp_criterion.manifold.expmap0(hyp_criterion.prototypes)
                dists = hyp_criterion.manifold.dist(x_hyp.unsqueeze(1), proto_hyp.unsqueeze(0))
                
                logits = -dists
                logits = logits.view(imgs.shape[0], imgs.shape[2], imgs.shape[3], -1).permute(0, 3, 1, 2)
                preds = torch.argmax(logits, dim=1)
                val_dice_metric.update(preds, masks)
        
        avg_val_dice = val_dice_metric.compute().item()
        print(f"Train Loss: {train_loss/len(src_loader):.4f}, Val Dice: {avg_val_dice:.4f}")
        
        scheduler.step(avg_val_dice)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            # Save both model and criterion (for prototypes)
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyp_state_dict': hyp_criterion.state_dict()
            }, "checkpoints/best_model_hyperbolic.pth")
            print(f"Saved Best Hyperbolic Model with Val Dice: {avg_val_dice:.4f}")

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
