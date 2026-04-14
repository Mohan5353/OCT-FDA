import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms
from fda import Feature_FDA

class MultiScaleFeatureFDAModel(nn.Module):
    def __init__(self, base_model, fda_L=0.05):
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head
        self.fda_L = fda_L

    def forward(self, x_src, x_trg=None):
        feats_src = list(self.encoder(x_src))
        
        if x_trg is not None:
            with torch.no_grad():
                feats_trg = self.encoder(x_trg)
            
            # Apply FDA at every level of the encoder pyramid
            for i in range(len(feats_src)):
                f_src = feats_src[i]
                f_trg = feats_trg[i]
                
                # Match batch size if needed
                if f_src.shape[0] > f_trg.shape[0]:
                    f_trg = f_trg.repeat(f_src.shape[0] // f_trg.shape[0] + 1, 1, 1, 1)[:f_src.shape[0]]
                else:
                    f_trg = f_trg[:f_src.shape[0]]
                
                # Apply FDA (lower L for shallow layers to preserve edges)
                L_i = self.fda_L if i > 2 else self.fda_L * 0.5
                feats_src[i] = Feature_FDA(f_src, f_trg, L=L_i)
            
        dec_out = self.decoder(feats_src)
        return self.segmentation_head(dec_out)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Multi-Scale Feature FDA Training on {device}")

    # Datasets
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Base Model
    base_model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    )

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        base_model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_path}")
        
    model = MultiScaleFeatureFDAModel(base_model, fda_L=args.fda_L).to(device)

    # Losses
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc=f"MS-FDA-Epoch {epoch+1}")
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
            outputs = model(src_imgs, trg_imgs)
            loss = seg_criterion(outputs, src_masks) + dice_loss(outputs, src_masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_multiscale_fda.pth")
            print(f"Saved Best MS-FDA Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fda_L", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
