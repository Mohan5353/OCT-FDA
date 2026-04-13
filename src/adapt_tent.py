import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
import copy

from dataset import RETOUCHDataset, get_val_transforms

def entropy_loss(logits):
    """
    TENT Entropy Loss: Minimizes Shannon Entropy to increase prediction confidence.
    """
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()

def setup_tent_model(model):
    """
    Configure model for Test-Time Adaptation (TENT).
    Only update Batch Normalization affine parameters (gamma and beta).
    """
    model.train() # TENT requires train mode to update BN running stats
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradients ONLY for BatchNorm parameters
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test-Time Adaptation (TENT) on {device}")

    # ONLY Target Data
    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=4
    ).to(device)

    model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    print(f"Loaded Base Model from {args.pretrained_path}")

    # TENT updates the model per batch. We need to save the final predictions.
    model = setup_tent_model(model)
    
    # Collect trainable params (BN gamma/beta)
    params, param_names = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
            param_names.append(name)
            
    optimizer = optim.Adam(params, lr=args.lr)
    
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)
    
    # Episodic TENT: We adapt and evaluate simultaneously per batch
    pbar = tqdm(trg_loader, desc="TENT Adaptation")
    for batch in pbar:
        imgs = batch['image'].to(device)
        masks = batch['mask'].to(device).long()
        
        # 1. Forward Pass & TENT Optimization
        optimizer.zero_grad()
        outputs = model(imgs)
        
        loss = entropy_loss(outputs)
        loss.backward()
        optimizer.step()
        
        # 2. Evaluate updated predictions on current batch
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            val_dice_metric.update(preds, masks)
            
        pbar.set_postfix({'ent_loss': f"{loss.item():.4f}"})
        
    avg_val_dice = val_dice_metric.compute().item()
    print(f"Final TENT Target Dice: {avg_val_dice:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/best_model_tent.pth")
    print(f"Saved Adapted TENT Model to checkpoints/best_model_tent.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=1) # TENT usually runs with batch=1 or very small batches
    parser.add_argument("--lr", type=float, default=1e-3) # TENT learning rate
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    args = parser.parse_args()
    main(args)
