import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from dataset import RETOUCHDataset, get_val_transforms
from hyperbolic import PoincareManifold

def evaluate_model(model, loader, device, hyp_criterion=None, embedding_dim=16, num_classes=4):
    model.eval()
    dice_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    dice_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    
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
            else:
                preds = torch.argmax(outputs[:, :num_classes, :, :], dim=1)
            
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
        num_classes_out = checkpoint['segmentation_head.0.weight'].shape[0]
        model = smp.Unet(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=num_classes_out).to(device)
        model.load_state_dict(checkpoint)
        print(f"Loaded standard checkpoint with {num_classes_out} output channels.")
        results = evaluate_model(model, loader, device, num_classes=4)
    
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
    parser.add_argument("--hyperbolic", action="store_true")
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--embedding_dim", type=int, default=16)
    args = parser.parse_args()
    main(args)
