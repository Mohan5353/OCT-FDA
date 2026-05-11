import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import RETOUCHDataset, get_val_transforms
from models.anamnet import AnamNet
from models.segresnet import SegResNet
from models.missformer import MISSFormer
from models.tiny_unet import TinyUnet

def get_model(model_name):
    if model_name == "anamnet": return AnamNet()
    if model_name == "segresnet": return SegResNet()
    if model_name == "missformer": return MISSFormer()
    if model_name == "tiny_unet": return TinyUnet()
    if model_name == "resnet101": 
        return smp.Unet(encoder_name="resnet101", in_channels=3, classes=4)
    if model_name == "resnet50":
        return smp.Unet(encoder_name="resnet50", in_channels=3, classes=4)
    if model_name == "resnet18":
        return smp.Unet(encoder_name="resnet18", in_channels=3, classes=4)
    if model_name == "resnet10":
        return smp.Unet(encoder_name="tu-resnet10t", in_channels=3, classes=4)
    if model_name == "mobilenet_v2":
        return smp.Unet(encoder_name="mobilenet_v2", in_channels=3, classes=4)
    if model_name == "convnext_large":
        return smp.Unet(encoder_name="tu-convnext_large", in_channels=3, classes=4)
    if model_name == "convnext_tiny":
        return smp.Unet(encoder_name="tu-convnext_tiny", in_channels=3, classes=4)
    if model_name == "convnext_atto":
        return smp.Unet(encoder_name="tu-convnext_atto", in_channels=3, classes=4)
    # Default to resnet101 for "best_model_*.pth" style
    return smp.Unet(encoder_name="resnet101", in_channels=3, classes=4)

def parse_checkpoint_name(ckpt_name):
    name = os.path.basename(ckpt_name).replace(".pth", "")
    parts = name.split("_")
    if len(parts) >= 3:
        if parts[1] == "model":
            method = "_".join(parts[2:])
            return "resnet101", method
        else:
            model = parts[1]
            if model == "convnext":
                model = f"{parts[1]}_{parts[2]}"
                mode = "_".join(parts[3:])
            else:
                mode = "_".join(parts[2:])
            return model, mode
    return "resnet101", "unknown"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = "samples"
    os.makedirs(output_root, exist_ok=True)
    
    img_size = (256, 256)
    dataset = RETOUCHDataset("data", vendor="Spectralis", transforms=get_val_transforms(img_size=img_size), load_mask=True, split='all')
    indices = [10, 50, 100, 200, 500]
    
    checkpoints = [os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pth")]
    
    for ckpt in tqdm(checkpoints, desc="Collecting Samples"):
        model_name, method_name = parse_checkpoint_name(ckpt)
        folder_name = f"{model_name}_{method_name}"
        save_dir = os.path.join(output_root, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            model = get_model(model_name).to(device)
            state_dict = torch.load(ckpt, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("base_model.", "").replace("module.", "")
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            
            with torch.no_grad():
                for idx in indices:
                    sample = dataset[idx]
                    img = sample['image'].unsqueeze(0).to(device)
                    mask = sample['mask'].numpy()
                    
                    output = model(img)
                    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                    
                    # Grayscale visualization
                    input_img_norm = sample['image'].permute(1, 2, 0).numpy()
                    input_img_gray = np.mean(input_img_norm, axis=2)
                    input_img_gray = (input_img_gray - input_img_gray.min()) / (input_img_gray.max() - input_img_gray.min() + 1e-6)
                    input_vis = (input_img_gray * 255).astype(np.uint8)
                    input_vis = cv2.cvtColor(input_vis, cv2.COLOR_GRAY2BGR)
                    
                    mask_vis = (mask * 80).astype(np.uint8)
                    mask_vis[mask == 3] = 255
                    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                    
                    pred_vis = (pred * 80).astype(np.uint8)
                    pred_vis[pred == 3] = 255
                    pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
                    
                    combined = np.hstack([input_vis, mask_vis, pred_vis])
                    cv2.imwrite(os.path.join(save_dir, f"sample_{idx}.png"), combined)
                    
        except Exception as e:
            print(f"Error processing {ckpt}: {e}")
            continue

if __name__ == "__main__":
    main()
