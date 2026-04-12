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

class EdgeGuidedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.tversky = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass')
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.edge_bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets, edge_preds, edge_targets):
        # preds: [B, 4, H, W], targets: [B, H, W]
        # edge_preds: [B, 1, H, W], edge_targets: [B, H, W]
        seg_loss = 0.5 * self.tversky(preds, targets) + 0.5 * self.lovasz(preds, targets) + self.ce(preds, targets)
        edge_loss = self.edge_bce(edge_preds.squeeze(1), edge_targets)
        return seg_loss + 0.5 * edge_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Edge-Guided Training on {device}")

    # 1. Datasets and Loaders
    # Source (Cirrus) with Edges
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='train', load_edge=True)
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Target (Spectralis) for FDA
    if args.use_fda:
        trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all', load_edge=False)
        trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        trg_loader = None

    # Val (Spectralis)
    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model: 5 output channels (4 fluid + 1 edge)
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5 
    ).to(device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        # Careful: pretrained weights might have 4 classes, we have 5
        state_dict = torch.load(args.pretrained_path, map_location=device)
        if state_dict['segmentation_head.0.weight'].shape[0] == 4:
            print("Adapting pretrained head from 4 to 5 classes...")
            # Initialize new head randomly, keep old classes
            old_weight = state_dict['segmentation_head.0.weight']
            old_bias = state_dict['segmentation_head.0.bias']
            new_weight = torch.randn((5, old_weight.shape[1], 3, 3)) * 0.01
            new_bias = torch.zeros(5)
            new_weight[:4] = old_weight
            new_bias[:4] = old_bias
            state_dict['segmentation_head.0.weight'] = new_weight
            state_dict['segmentation_head.0.bias'] = new_bias
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {args.pretrained_path}")

    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    criterion = EdgeGuidedLoss(class_weights)
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
        trg_iter = iter(trg_loader) if args.use_fda else None
        
        pbar = tqdm(src_loader, desc="Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            src_edges = batch['edge'].to(device).float()
            
            if args.use_fda:
                try:
                    trg_batch = next(trg_iter)
                except (StopIteration, TypeError):
                    trg_iter = iter(trg_loader)
                    trg_batch = next(trg_iter)
                
                trg_imgs = trg_batch['image'].to(device)
                # FDA style transfer
                if trg_imgs.shape[0] < src_imgs.shape[0]:
                    src_imgs_adapted = FDA_source_to_target(src_imgs[:trg_imgs.shape[0]], trg_imgs, L=args.fda_L)
                    src_imgs = torch.cat([src_imgs_adapted, src_imgs[trg_imgs.shape[0]:]], dim=0)
                else:
                    src_imgs = FDA_source_to_target(src_imgs, trg_imgs[:src_imgs.shape[0]], L=args.fda_L)

            optimizer.zero_grad()
            outputs = model(src_imgs)
            
            # Split outputs: [B, 0:4] for fluids, [B, 4] for edge
            fluid_preds = outputs[:, :4, :, :]
            edge_preds = outputs[:, 4:5, :, :]
            
            loss = criterion(fluid_preds, src_masks, edge_preds, src_edges)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation (only on fluid classes)
        model.eval()
        val_dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device).long()
                outputs = model(imgs)
                fluid_preds = outputs[:, :4, :, :]
                preds = torch.argmax(fluid_preds, dim=1)
                val_dice_metric.update(preds, masks)
        
        avg_val_dice = val_dice_metric.compute().item()
        print(f"Train Loss: {train_loss/len(src_loader):.4f}, Val Dice: {avg_val_dice:.4f}")
        
        scheduler.step(avg_val_dice)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model_edge_guided.pth")
            print(f"Saved Best Edge-Guided Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_fda", action="store_true", help="Enable Fourier Domain Adaptation")
    parser.add_argument("--fda_L", type=float, default=0.05, help="FDA mutation ratio")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
