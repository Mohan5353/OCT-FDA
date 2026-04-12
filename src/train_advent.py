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
from advent import Discriminator, prob_2_entropy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ADVENT (Adversarial Entropy Minimization) Training on {device}")

    # 1. Datasets and Loaders
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(), load_mask=True, split='all')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Models
    # Segmenter
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    ).to(device)

    # Discriminator
    discr = Discriminator(num_classes=4).to(device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        # Handle dict or direct state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {args.pretrained_path}")

    # 3. Optimizers & Losses
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    tversky_loss = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7)
    lovasz_loss = smp.losses.LovaszLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def seg_criterion(preds, targets):
        return 0.5 * tversky_loss(preds, targets) + 0.5 * lovasz_loss(preds, targets) + ce_loss(preds, targets)

    bce_loss = nn.BCEWithLogitsLoss()

    optimizer_seg = optim.Adam(model.parameters(), lr=args.lr_seg)
    optimizer_dis = optim.Adam(discr.parameters(), lr=args.lr_dis)
    
    # 4. Labels for Adversarial Loss
    source_label = 0
    target_label = 1

    # 5. Training Loop
    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        discr.train()
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc="ADVENT-Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            trg_imgs = trg_batch['image'].to(device)

            # --- 1. Train Segmenter (Model) ---
            optimizer_seg.zero_grad()
            optimizer_dis.zero_grad() # Keep discr weights fixed for this part
            
            # FDA Augmentation (optional, we'll use it to help)
            if args.use_fda:
                if trg_imgs.shape[0] < src_imgs.shape[0]:
                    src_imgs_fda = torch.cat([FDA_source_to_target(src_imgs[:trg_imgs.shape[0]], trg_imgs, L=args.fda_L), 
                                              src_imgs[trg_imgs.shape[0]:]], dim=0)
                else:
                    src_imgs_fda = FDA_source_to_target(src_imgs, trg_imgs[:src_imgs.shape[0]], L=args.fda_L)
                src_outputs = model(src_imgs_fda)
            else:
                src_outputs = model(src_imgs)
                
            loss_seg = seg_criterion(src_outputs, src_masks)
            loss_seg.backward()
            
            # Adversarial Step (Segmenter tries to fool Discriminator on Target)
            trg_outputs = model(trg_imgs)
            trg_probs = torch.softmax(trg_outputs, dim=1)
            trg_entropy = prob_2_entropy(trg_probs)
            
            dis_outputs_trg = discr(trg_entropy)
            # Fools discriminator by pretending target is source
            loss_adv = bce_loss(dis_outputs_trg, torch.FloatTensor(dis_outputs_trg.size()).fill_(source_label).to(device))
            
            loss_total_seg = args.lambda_adv * loss_adv
            loss_total_seg.backward()
            
            optimizer_seg.step()

            # --- 2. Train Discriminator ---
            optimizer_dis.zero_grad()
            
            # Detach to prevent gradients from flowing back to segmenter
            src_probs = torch.softmax(src_outputs.detach(), dim=1)
            src_entropy = prob_2_entropy(src_probs)
            
            # Source Forward
            dis_outputs_src = discr(src_entropy)
            loss_dis_src = bce_loss(dis_outputs_src, torch.FloatTensor(dis_outputs_src.size()).fill_(source_label).to(device))
            
            # Target Forward
            trg_entropy_dis = trg_entropy.detach()
            dis_outputs_trg = discr(trg_entropy_dis)
            loss_dis_trg = bce_loss(dis_outputs_trg, torch.FloatTensor(dis_outputs_trg.size()).fill_(target_label).to(device))
            
            loss_dis = (loss_dis_src + loss_dis_trg) / 2
            loss_dis.backward()
            optimizer_dis.step()

            pbar.set_postfix({'seg_l': f"{loss_seg.item():.3f}", 'adv_l': f"{loss_adv.item():.3f}", 'dis_l': f"{loss_dis.item():.3f}"})
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_advent.pth")
            print(f"Saved Best ADVENT Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr_seg", type=float, default=5e-5)
    parser.add_argument("--lr_dis", type=float, default=1e-4)
    parser.add_argument("--lambda_adv", type=float, default=0.01) # Adverarial loss weight
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_fda", action="store_true", help="Combine ADVENT with FDA style transfer")
    parser.add_argument("--fda_L", type=float, default=0.05)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    args = parser.parse_args()
    main(args)
