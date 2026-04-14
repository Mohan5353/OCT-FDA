import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

from dataset import RETOUCHDataset, get_train_transforms, get_val_transforms
from fda import Advanced_Feature_FDA
from advanced_losses import MIEstimator, mi_loss, PhysicsAttenuationLoss, TopologicalLoss

class AdvancedFeatureFDAModel(nn.Module):
    def __init__(self, base_model, fda_L=0.05):
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head
        self.fda_L = fda_L

    def forward(self, x_src, x_trg=None):
        feats_src = self.encoder(x_src)
        
        amp = None
        pha = None
        mutated_feat = None

        if x_trg is not None:
            with torch.no_grad():
                feats_trg = self.encoder(x_trg)
            
            # Apply FDA on the bottleneck features (the last feature map)
            bottleneck_src = feats_src[-1]
            bottleneck_trg = feats_trg[-1]
            
            # Match batch size if needed
            if bottleneck_src.shape[0] > bottleneck_trg.shape[0]:
                b_trg = bottleneck_trg.repeat(bottleneck_src.shape[0] // bottleneck_trg.shape[0] + 1, 1, 1, 1)[:bottleneck_src.shape[0]]
            else:
                b_trg = bottleneck_trg[:bottleneck_src.shape[0]]
                
            mutated_bottleneck, amp, pha = Advanced_Feature_FDA(bottleneck_src, b_trg, L=self.fda_L)
            mutated_feat = mutated_bottleneck
            
            feats_src = list(feats_src)
            feats_src[-1] = mutated_bottleneck
            
        dec_out = self.decoder(feats_src)
        preds = self.segmentation_head(dec_out)
        
        if x_trg is not None:
            return preds, amp, pha, mutated_feat
        return preds

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Advanced Feature-Space FDA Training on {device}")

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
        
    model = AdvancedFeatureFDAModel(base_model, fda_L=args.fda_L).to(device)

    # MI Estimator
    # Bottleneck of ResNet101 has 2048 channels
    mi_estimator = MIEstimator(channels=2048, feature_size=None).to(device)

    # Losses
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    
    physics_loss_fn = PhysicsAttenuationLoss().to(device)
    topo_loss_fn = TopologicalLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        mi_estimator.train()
        
        train_loss = 0
        trg_iter = iter(trg_loader)
        
        pbar = tqdm(src_loader, desc="Adv-FDA-Train")
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            
            try:
                trg_batch = next(trg_iter)
            except (StopIteration, TypeError):
                trg_iter = iter(trg_loader)
                trg_batch = next(trg_iter)
            trg_imgs = trg_batch['image'].to(device)

            # 1. Update MI Estimator (Adversarial style or simple estimation)
            optimizer.zero_grad()
            mi_optimizer.zero_grad()
            
            outputs, amp, pha, mutated_feat = model(src_imgs, trg_imgs)
            
            # Loss terms
            loss_seg = seg_criterion(outputs, src_masks) + dice_loss(outputs, src_masks)
            
            # 1. MI Regularization
            loss_mi = mi_loss(mi_estimator, amp, pha)
            
            # 2. Physics Regularization
            loss_phys = physics_loss_fn(mutated_feat)
            
            # 3. Topological Regularization
            loss_topo = topo_loss_fn(outputs, src_masks)
            
            total_loss = loss_seg + args.w_mi * loss_mi + args.w_phys * loss_phys + args.w_topo * loss_topo
            
            total_loss.backward()
            optimizer.step()
            mi_optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({
                'L_seg': f"{loss_seg.item():.3f}",
                'L_mi': f"{loss_mi.item():.3f}",
                'L_phys': f"{loss_phys.item():.3f}",
                'L_topo': f"{loss_topo.item():.3f}"
            })
            
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
            torch.save(model.state_dict(), "checkpoints/best_model_advanced_feature_fda.pth")
            print(f"Saved Best Advanced Feature FDA Model with Val Dice: {avg_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fda_L", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/best_model_baseline.pth")
    
    # Weights for regularizers
    parser.add_argument("--w_mi", type=float, default=0.1, help="Weight for Phase-Amplitude MI minimization")
    parser.add_argument("--w_phys", type=float, default=0.01, help="Weight for Physics-Informed Attenuation")
    parser.add_argument("--w_topo", type=float, default=0.01, help="Weight for Topological Connectivity")
    
    args = parser.parse_args()
    main(args)
