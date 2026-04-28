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
from fda import Feature_FDA, Advanced_Feature_FDA, Distribution_Feature_FDA
from advanced_losses import MIEstimator, mi_loss, PhysicsAttenuationLoss, TopologicalLoss
from ddsp import DistributionDisruptionModule, disruption_consistency_loss
from dann_modules import DomainDiscriminator
from models.anamnet import AnamNet
from models.segresnet import SegResNet
from models.missformer import MISSFormer

class CustomFlexibleUDAModel(nn.Module):
    def __init__(self, base_model, mode='baseline', fda_L=0.01):
        super().__init__()
        self.base_model = base_model
        self.mode = mode
        self.fda_L = fda_L

    def forward(self, x_src, x_trg=None, ddm=None, alpha=None, discr=None):
        if hasattr(self.base_model, 'get_encoder_features'):
            feats_src = list(self.base_model.get_encoder_features(x_src))
        else:
            feats_src = list(self.base_model.encoder(x_src))
            
        amp, pha, mutated_feat = None, None, None
        loss_cons = None
        loss_domain = None

        if x_trg is not None and self.mode != 'baseline':
            with torch.no_grad():
                if hasattr(self.base_model, 'get_encoder_features'):
                    feats_trg = self.base_model.get_encoder_features(x_trg)
                else:
                    feats_trg = self.base_model.encoder(x_trg)
            
            if self.mode == 'fda':
                idx = -1
                f_src, f_trg = feats_src[idx], feats_trg[idx]
                if f_src.shape[0] > f_trg.shape[0]:
                    f_trg = f_trg.repeat(f_src.shape[0] // f_trg.shape[0] + 1, 1, 1, 1)[:f_src.shape[0]]
                else: f_trg = f_trg[:f_src.shape[0]]
                feats_src[idx] = Feature_FDA(f_src, f_trg, L=self.fda_L)

            elif self.mode == 'ms-fda':
                for i in range(len(feats_src)):
                    f_src, f_trg = feats_src[i], feats_trg[i]
                    if f_src.shape[1] == 0: continue
                    if f_src.shape[0] > f_trg.shape[0]:
                        f_trg = f_trg.repeat(f_src.shape[0] // f_trg.shape[0] + 1, 1, 1, 1)[:f_src.shape[0]]
                    else: f_trg = f_trg[:f_src.shape[0]]
                    L_i = self.fda_L if i > (len(feats_src)//2) else self.fda_L * 0.5
                    feats_src[i] = Feature_FDA(f_src, f_trg, L=L_i)

            elif self.mode == 'adv-fda':
                idx = -1
                f_src, f_trg = feats_src[idx], feats_trg[idx]
                mutated_feat, amp, pha = Distribution_Feature_FDA(f_src, f_trg, L=self.fda_L)
                feats_src[idx] = mutated_feat

            elif self.mode == 'adv-1to1':
                idx = -1
                f_src, f_trg = feats_src[idx], feats_trg[idx]
                if f_src.shape[0] > f_trg.shape[0]:
                    f_trg = f_trg.repeat(f_src.shape[0] // f_trg.shape[0] + 1, 1, 1, 1)[:f_src.shape[0]]
                else: f_trg = f_trg[:f_src.shape[0]]
                mutated_feat, amp, pha = Advanced_Feature_FDA(f_src, f_trg, L=self.fda_L)
                feats_src[idx] = mutated_feat

            elif self.mode == 'ddsp' and ddm is not None:
                # Disrupt the second level features
                idx = 1
                f_src, f_trg = feats_src[idx], feats_trg[idx]
                
                # Match batch size for disruption module
                if f_src.shape[0] > f_trg.shape[0]:
                    f_trg = f_trg.repeat(f_src.shape[0] // f_trg.shape[0] + 1, 1, 1, 1)[:f_src.shape[0]]
                else: f_trg = f_trg[:f_src.shape[0]]
                
                src_disrupted, _ = ddm(f_src, f_trg)
                
                # We need consistent prediction for consistency loss
                # Original prediction
                if hasattr(self.base_model, 'forward_from_features'):
                    preds_orig = self.base_model.forward_from_features(feats_src)
                else:
                    preds_orig = self.base_model.segmentation_head(self.base_model.decoder(feats_src))
                
                # Disrupted prediction
                feats_src_dis = list(feats_src)
                feats_src_dis[idx] = src_disrupted
                if hasattr(self.base_model, 'forward_from_features'):
                    preds_dis = self.base_model.forward_from_features(feats_src_dis)
                else:
                    preds_dis = self.base_model.segmentation_head(self.base_model.decoder(feats_src_dis))
                
                return preds_orig, preds_dis # Return both for consistency loss

            elif self.mode == 'dann' and discr is not None and alpha is not None:
                # Get bottleneck pooled features for discriminator
                src_bottleneck = feats_src[-1]
                trg_bottleneck = feats_trg[-1]
                src_pooled = torch.mean(src_bottleneck, dim=(2, 3))
                trg_pooled = torch.mean(trg_bottleneck, dim=(2, 3))
                
                src_domain_preds = discr(src_pooled, alpha=alpha)
                trg_domain_preds = discr(trg_pooled, alpha=alpha)
                
                if hasattr(self.base_model, 'forward_from_features'):
                    preds = self.base_model.forward_from_features(feats_src)
                else:
                    preds = self.base_model.segmentation_head(self.base_model.decoder(feats_src))
                
                return preds, src_domain_preds, trg_domain_preds
            
        if hasattr(self.base_model, 'forward_from_features'):
            preds = self.base_model.forward_from_features(feats_src)
        else:
            preds = self.base_model.segmentation_head(self.base_model.decoder(feats_src))
            
        if (self.mode == 'adv-fda' or self.mode == 'adv-1to1') and x_trg is not None:
            return preds, amp, pha, mutated_feat
        return preds

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {args.model_name} | Mode: {args.mode} | Image Size: {args.img_size}")

    if args.model_name == "anamnet": base_model = AnamNet()
    elif args.model_name == "segresnet": base_model = SegResNet()
    elif args.model_name == "missformer": base_model = MISSFormer()
    elif args.model_name == "resnet101": 
        base_model = smp.Unet(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=4)
    elif args.model_name == "resnet50":
        base_model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=4)
    elif args.model_name == "resnet18":
        base_model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=4)
    elif args.model_name == "convnext_large":
        base_model = smp.Unet(encoder_name="tu-convnext_large", encoder_weights="imagenet", in_channels=3, classes=4)
    else: raise ValueError("Unknown model name")

    model = CustomFlexibleUDAModel(base_model, mode=args.mode, fda_L=args.fda_L).to(device)

    # Specialized Modules & Losses
    mi_estimator, mi_optimizer = None, None
    ddm = None
    discr, discr_optimizer = None, None
    bce_loss = nn.BCEWithLogitsLoss()

    if args.mode in ['adv-fda', 'adv-1to1']:
        dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        with torch.no_grad():
            if hasattr(base_model, 'get_encoder_features'):
                bn_channels = base_model.get_encoder_features(dummy)[-1].shape[1]
            else:
                bn_channels = base_model.encoder(dummy)[-1].shape[1]
        mi_estimator = MIEstimator(channels=bn_channels, feature_size=None).to(device)
        mi_optimizer = optim.Adam(mi_estimator.parameters(), lr=args.lr)
    
    elif args.mode == 'ddsp':
        ddm = DistributionDisruptionModule(p=0.5).to(device)
    
    elif args.mode == 'dann':
        dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        with torch.no_grad():
            if hasattr(base_model, 'get_encoder_features'):
                bn_channels = base_model.get_encoder_features(dummy)[-1].shape[1]
            else:
                bn_channels = base_model.encoder(dummy)[-1].shape[1]
        discr = DomainDiscriminator(input_dim=bn_channels).to(device)
        discr_optimizer = optim.Adam(discr.parameters(), lr=args.lr)

    # Datasets
    img_size = (args.img_size, args.img_size)
    src_ds = RETOUCHDataset(args.data_root, vendor="Cirrus", transforms=get_train_transforms(img_size=img_size), load_mask=True, split='all')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    trg_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_train_transforms(img_size=img_size), load_mask=False, split='all')
    trg_loader = DataLoader(trg_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_ds = RETOUCHDataset(args.data_root, vendor="Spectralis", transforms=get_val_transforms(img_size=img_size), load_mask=True, split='all')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    class_weights = torch.tensor([1.0, 50.0, 50.0, 100.0]).to(device)
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass')
    phys_loss_fn = PhysicsAttenuationLoss().to(device) if args.mode in ['adv-fda', 'adv-1to1'] else None
    topo_loss_fn = TopologicalLoss().to(device) if args.mode in ['adv-fda', 'adv-1to1'] else None

    best_val_dice = 0.0
    from torchmetrics.classification import MulticlassF1Score
    val_dice_metric = MulticlassF1Score(num_classes=4).to(device)

    num_steps = args.epochs * len(src_loader)
    current_step = 0

    for epoch in range(args.epochs):
        model.train()
        if mi_estimator: mi_estimator.train()
        if discr: discr.train()
        trg_iter = iter(trg_loader)
        pbar = tqdm(src_loader, desc=f"{args.model_name}-{args.mode}-E{epoch+1}")
        
        for batch in pbar:
            src_imgs = batch['image'].to(device)
            src_masks = batch['mask'].to(device).long()
            try: trg_imgs = next(trg_iter)['image'].to(device)
            except: 
                trg_iter = iter(trg_loader)
                trg_imgs = next(trg_iter)['image'].to(device)
            
            optimizer.zero_grad()
            if mi_optimizer: mi_optimizer.zero_grad()
            if discr_optimizer: discr_optimizer.zero_grad()
            
            if args.mode in ['adv-fda', 'adv-1to1']:
                outputs, amp, pha, mutated_feat = model(src_imgs, trg_imgs)
                l_seg = seg_criterion(outputs, src_masks) + dice_loss_fn(outputs, src_masks)
                l_mi = torch.clamp(mi_loss(mi_estimator, amp, pha), min=0.0)
                l_phys = phys_loss_fn(mutated_feat)
                l_topo = topo_loss_fn(outputs, src_masks)
                loss = l_seg + args.w_mi * l_mi + args.w_phys * l_phys + args.w_topo * l_topo
                loss.backward()
                optimizer.step()
                if mi_optimizer: mi_optimizer.step()
                
            elif args.mode == 'ddsp':
                preds_orig, preds_dis = model(src_imgs, trg_imgs, ddm=ddm)
                l_seg = seg_criterion(preds_orig, src_masks) + dice_loss_fn(preds_orig, src_masks)
                l_cons = disruption_consistency_loss(preds_orig, preds_dis)
                loss = l_seg + 1.0 * l_cons
                loss.backward()
                optimizer.step()

            elif args.mode == 'dann':
                p = float(current_step) / num_steps
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                current_step += 1
                preds, src_dom, trg_dom = model(src_imgs, trg_imgs, alpha=alpha, discr=discr)
                l_seg = seg_criterion(preds, src_masks) + dice_loss_fn(preds, src_masks)
                l_dom = bce_loss(src_dom, torch.zeros_like(src_dom)) + bce_loss(trg_dom, torch.ones_like(trg_dom))
                loss = l_seg + 0.1 * l_dom
                loss.backward()
                optimizer.step()
                if discr_optimizer: discr_optimizer.step()
            
            else:
                outputs = model(src_imgs, trg_imgs)
                loss = seg_criterion(outputs, src_masks) + dice_loss_fn(outputs, src_masks)
                loss.backward()
                optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})

        # Validation
        model.eval()
        val_dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device).long()
                outputs = model(imgs) 
                preds = torch.argmax(outputs, dim=1)
                val_dice_metric.update(preds, masks)
        avg_val_dice = val_dice_metric.compute().item()
        print(f"Val Dice: {avg_val_dice:.4f}")
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.base_model.state_dict(), f"checkpoints/best_{args.model_name}_{args.mode}.pth")
            print(f"Saved Best {args.model_name} {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["anamnet", "segresnet", "missformer", "resnet101", "resnet50", "resnet18", "convnext_large"])
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "fda", "ms-fda", "adv-fda", "adv-1to1", "ddsp", "dann"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fda_L", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--w_mi", type=float, default=1e-6)
    parser.add_argument("--w_phys", type=float, default=0.01)
    parser.add_argument("--w_topo", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
