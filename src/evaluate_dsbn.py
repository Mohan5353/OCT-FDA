import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from dataset import RETOUCHDataset, get_val_transforms
from dsbn import convert_dsbn, set_model_domain

def evaluate_model(model, loader, device, domain_idx=None):
    model.eval()
    if domain_idx is not None:
        set_model_domain(model, domain_idx)
        
    dice_metric = MulticlassF1Score(num_classes=4).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=4).to(device)
    dice_per_class = MulticlassF1Score(num_classes=4, average=None).to(device)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            
            outputs = model(imgs)
            preds = torch.argmax(outputs[:, :4, :, :], dim=1)
            
            dice_metric.update(preds, masks)
            iou_metric.update(preds, masks)
            dice_per_class.update(preds, masks)
            
    return {
        'dice': dice_metric.compute().item(),
        'iou': iou_metric.compute().item(),
        'dice_per_class': dice_per_class.compute().cpu().numpy()
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    
    ds = RETOUCHDataset(args.data_root, vendor=args.vendor, transforms=get_val_transforms(), load_mask=True, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Setup model
    # We must use same base as training
    model = smp.Unet(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=4).to(device)
    
    if args.dsbn:
        convert_dsbn(model, num_domains=2)
        model.to(device)
        print("Converted model to DSBN for evaluation.")
        domain_idx = 1 if args.vendor == "Spectralis" else 0
    else:
        domain_idx = None

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    results = evaluate_model(model, loader, device, domain_idx=domain_idx)
    
    print("\n" + "="*30)
    print(f"  Results for {args.vendor} ({args.split})")
    print("="*30)
    print(f"Overall Dice: {results['dice']:.4f}")
    print(f"Overall IoU:  {results['iou']:.4f}")
    fluid_names = ["Background", "IRF", "SRF", "PED"]
    for i, dice in enumerate(results['dice_per_class']):
        print(f"  {fluid_names[i]:10}: {dice:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--vendor", type=str, default="Spectralis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dsbn", action="store_true")
    args = parser.parse_args()
    main(args)
