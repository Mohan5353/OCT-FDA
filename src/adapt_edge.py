import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms

def entropy_loss(preds):
    """
    Minimizes Shannon Entropy to increase prediction confidence on target data.
    """
    probs = torch.softmax(preds, dim=1)
    log_probs = torch.log_softmax(preds, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Source-Free Domain Adaptation (SFDA) on {device}")

    # ONLY Target Data - no Source dataset loaded
    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=4
    ).to(device)

    model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    print(f"Loaded Base Model from {args.pretrained_path}")

    # Low learning rate for delicate fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(trg_loader, desc=f"SFDA Epoch {epoch+1}")
        for batch in pbar:
            imgs = batch['image'].to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Label-free entropy loss
            loss = entropy_loss(outputs)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'ent_loss': f"{loss.item():.4f}"})
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_sfda.pth")
            print(f"Saved Best SFDA Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    args = parser.parse_args()
    main(args)
