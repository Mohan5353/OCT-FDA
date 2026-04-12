import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import tifffile as tiff
import numpy as np

from dataset import RETOUCHDataset, get_val_transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "data"
    checkpoint_path = "checkpoints/best_model_baseline.pth"
    output_dir = "data/Spectralis/pseudo_masks_filtered"
    CONF_THRESHOLD = 0.95
    IGNORE_INDEX = 255
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=4
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded baseline model from {checkpoint_path}")

    # 2. Target Dataset (Spectralis)
    ds = RETOUCHDataset(data_root, vendor="Spectralis", transforms=get_val_transforms(), load_mask=False, split='all')
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # 3. Generate and Save Filtered Pseudo-labels
    print(f"Generating filtered pseudo-labels (Threshold: {CONF_THRESHOLD})...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            img = batch['image'].to(device)
            img_path = ds.image_paths[i]
            filename = os.path.basename(img_path)
            
            output = model(img)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            
            # Mask out low confidence pixels
            final_pred = pred.squeeze().cpu().numpy().astype(np.uint8)
            conf_mask = conf.squeeze().cpu().numpy() < CONF_THRESHOLD
            
            # Set low confidence pixels to IGNORE_INDEX
            final_pred[conf_mask] = IGNORE_INDEX
            
            # Save as TIFF
            save_path = os.path.join(output_dir, filename)
            tiff.imwrite(save_path, final_pred)

    print(f"Filtered pseudo-labels saved to {output_dir}")

if __name__ == "__main__":
    main()
