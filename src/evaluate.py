import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
import numpy as np
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from scipy.ndimage import label, distance_transform_edt
from dataset import RETOUCHDataset, get_val_transforms
from hyperbolic import PoincareManifold

def compute_ece(preds_probs, targets, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = np.zeros(preds_probs.shape[1])
    for c in range(1, preds_probs.shape[1]):
        confidences = preds_probs[:, c, ...].flatten()
        accuracies = (targets == c).flatten()
        
        ece_c = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_c += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        ece[c] = ece_c
    return ece

def compute_surface_distances(mask_gt, mask_pred, spacing=(1.0, 1.0)):
    if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0:
        return np.nan, np.nan
        
    border_gt = mask_gt ^ (distance_transform_edt(mask_gt) > 1)
    border_pred = mask_pred ^ (distance_transform_edt(mask_pred) > 1)
    
    dist_to_gt = distance_transform_edt(~border_gt, sampling=spacing)
    dist_to_pred = distance_transform_edt(~border_pred, sampling=spacing)
    
    distances_gt_to_pred = dist_to_pred[border_gt]
    distances_pred_to_gt = dist_to_gt[border_pred]
    
    sur_dists = np.concatenate([distances_gt_to_pred, distances_pred_to_gt])
    if len(sur_dists) == 0:
        return np.nan, np.nan
        
    hd95 = np.percentile(sur_dists, 95)
    assd = np.mean(sur_dists)
    return hd95, assd

def compute_lesion_detection(mask_gt, mask_pred):
    gt_labeled, num_gt = label(mask_gt)
    pred_labeled, num_pred = label(mask_pred)
    
    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, 1.0
    if num_gt == 0:
        return 0.0, 0.0, 0.0
    if num_pred == 0:
        return 0.0, 1.0, 0.0
        
    true_positives = 0
    for i in range(1, num_gt + 1):
        if np.any(mask_pred[gt_labeled == i]):
            true_positives += 1
            
    false_positives = 0
    for i in range(1, num_pred + 1):
        if not np.any(mask_gt[pred_labeled == i]):
            false_positives += 1
            
    recall = true_positives / num_gt
    precision = true_positives / (true_positives + false_positives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return recall, precision, f1

def evaluate_model(model, loader, device, hyp_criterion=None, embedding_dim=16, num_classes=4):
    model.eval()
    dice_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    dice_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    
    # Radiomics & Advanced Metrics
    gt_volumes = {i: [] for i in range(1, num_classes)}
    pred_volumes = {i: [] for i in range(1, num_classes)}
    
    all_ece = []
    all_hd95 = {i: [] for i in range(1, num_classes)}
    all_assd = {i: [] for i in range(1, num_classes)}
    all_lesion_f1 = {i: [] for i in range(1, num_classes)}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            
            outputs = model(imgs)
            
            if hyp_criterion is not None:
                # Hyperbolic mapping
                x_flat = outputs.permute(0, 2, 3, 1).reshape(-1, embedding_dim)
                x_hyp = hyp_criterion.manifold.expmap0(x_flat)
                proto_hyp = hyp_criterion.manifold.expmap0(hyp_criterion.prototypes)
                dists = hyp_criterion.manifold.dist(x_hyp.unsqueeze(1), proto_hyp.unsqueeze(0))
                logits = -dists
                logits = logits.view(imgs.shape[0], imgs.shape[2], imgs.shape[3], -1).permute(0, 3, 1, 2)
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
            else:
                preds = torch.argmax(outputs[:, :num_classes, :, :], dim=1)
                probs = torch.softmax(outputs[:, :num_classes, :, :], dim=1)
            
            dice_metric.update(preds, masks)
            iou_metric.update(preds, masks)
            dice_per_class.update(preds, masks)
            
            # ECE
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            ece = compute_ece(probs_np, masks_np, n_bins=10)
            all_ece.append(ece)
            
            preds_np = preds.cpu().numpy()
            
            for b in range(imgs.size(0)):
                for c in range(1, num_classes):
                    gt_c = (masks_np[b] == c)
                    pred_c = (preds_np[b] == c)
                    
                    gt_volumes[c].append(gt_c.sum())
                    pred_volumes[c].append(pred_c.sum())
                    
                    hd95, assd = compute_surface_distances(gt_c, pred_c)
                    if not np.isnan(hd95):
                        all_hd95[c].append(hd95)
                        all_assd[c].append(assd)
                        
                    _, _, lf1 = compute_lesion_detection(gt_c, pred_c)
                    all_lesion_f1[c].append(lf1)
            
    volume_mae = {}
    hd95_mean = {}
    assd_mean = {}
    lesion_f1_mean = {}
    for c in range(1, num_classes):
        mae = np.mean(np.abs(np.array(gt_volumes[c]) - np.array(pred_volumes[c])))
        volume_mae[c] = mae
        hd95_mean[c] = np.mean(all_hd95[c]) if len(all_hd95[c]) > 0 else np.nan
        assd_mean[c] = np.mean(all_assd[c]) if len(all_assd[c]) > 0 else np.nan
        lesion_f1_mean[c] = np.mean(all_lesion_f1[c]) if len(all_lesion_f1[c]) > 0 else 0.0
        
    mean_ece = np.mean(all_ece, axis=0)
        
    return {
        'dice': dice_metric.compute().item(),
        'iou': iou_metric.compute().item(),
        'dice_per_class': dice_per_class.compute().cpu().numpy(),
        'volume_mae': volume_mae,
        'ece': mean_ece,
        'hd95': hd95_mean,
        'assd': assd_mean,
        'lesion_f1': lesion_f1_mean
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    
    ds = RETOUCHDataset(args.data_root, vendor=args.vendor, transforms=get_val_transforms(), load_mask=True, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if args.hyperbolic:
        model = smp.Unet(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=args.embedding_dim).to(device)
        from hyperbolic import HyperbolicCrossEntropyLoss
        hyp_criterion = HyperbolicCrossEntropyLoss(num_classes=4, embedding_dim=args.embedding_dim)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        hyp_criterion.load_state_dict(checkpoint['hyp_state_dict'])
        hyp_criterion.to(device)
        print("Loaded Hyperbolic checkpoint.")
        results = evaluate_model(model, loader, device, hyp_criterion=hyp_criterion, embedding_dim=args.embedding_dim)
    elif args.transformer:
        model = smp.Segformer(encoder_name="mit_b3", encoder_weights=None, in_channels=3, classes=4).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print("Loaded SegFormer checkpoint.")
        results = evaluate_model(model, loader, device, num_classes=4)
    else:
        # Standard or Edge-Guided
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Check if 5-channel
        if 'segmentation_head.0.weight' in checkpoint:
            num_classes_out = checkpoint['segmentation_head.0.weight'].shape[0]
        else:
            num_classes_out = 4
        model = smp.Unet(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=num_classes_out).to(device)
        model.load_state_dict(checkpoint)
        print(f"Loaded standard checkpoint with {num_classes_out} output channels.")
        results = evaluate_model(model, loader, device, num_classes=4)
    
    print("\n" + "="*40)
    print(f"  Results for {args.vendor} ({args.split})")
    print("="*40)
    print(f"Overall Dice: {results['dice']:.4f}")
    print(f"Overall IoU:  {results['iou']:.4f}")
    fluid_names = ["Background", "IRF", "SRF", "PED"]
    print("\nDice per class:")
    for i, dice in enumerate(results['dice_per_class']):
        print(f"  {fluid_names[i]:10}: {dice:.4f}")
        
    print("\nExpected Calibration Error (ECE):")
    for i, ece in enumerate(results['ece']):
        if i > 0:
            print(f"  {fluid_names[i]:10}: {ece:.4f}")
            
    print("\nBoundary/Surface Distances:")
    for i in range(1, 4):
        print(f"  {fluid_names[i]:10}: HD95 = {results['hd95'][i]:.2f} px, ASSD = {results['assd'][i]:.2f} px")
        
    print("\nLesion-Wise Detection Rate (F1):")
    for i in range(1, 4):
        print(f"  {fluid_names[i]:10}: {results['lesion_f1'][i]:.4f}")
        
    if 'volume_mae' in results:
        print("\nRadiomics Quantification (Volume MAE in pixels):")
        for c in range(1, 4):
            print(f"  {fluid_names[c]:10}: {results['volume_mae'][c]:.2f}")
            
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--vendor", type=str, default="Spectralis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hyperbolic", action="store_true")
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--embedding_dim", type=int, default=16)
    args = parser.parse_args()
    main(args)
