import os
import glob
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RETOUCHDataset(Dataset):
    """
    Generic Dataset for OCT Fluid Segmentation from RETOUCH.
    Supports vendor-specific or full-dataset access with train/test splits.
    Now includes auxiliary edge map loading.
    """
    def __init__(self, data_root, vendor=None, transforms=None, load_mask=True, split='all', split_ratio=0.8, load_edge=False):
        """
        Args:
            data_root (str): Root directory of the dataset.
            vendor (str): 'Cirrus', 'Spectralis', or 'Topcon'.
            transforms (A.Compose): Albumentations transforms.
            load_mask (bool): Whether to load fluid masks.
            split (str): 'train', 'test', or 'all'.
            split_ratio (float): Ratio of volumes to use for training.
            load_edge (bool): Whether to load anatomical edge maps.
        """
        self.data_root = data_root
        self.vendor = vendor
        self.transforms = transforms
        self.load_mask = load_mask
        self.load_edge = load_edge

        # Find all images for the specified vendor
        search_pattern = os.path.join(data_root, vendor if vendor else "*", "cropped_images", "*.tiff")
        all_image_paths = sorted(glob.glob(search_pattern))
        
        # Get unique volumes (e.g., TRAIN001)
        volumes = sorted(list(set([os.path.basename(p).split('_')[1] for p in all_image_paths])))
        
        # Deterministic split
        num_train = int(len(volumes) * split_ratio)
        if split == 'train':
            selected_volumes = volumes[:num_train]
        elif split == 'test':
            selected_volumes = volumes[num_train:]
        else:
            selected_volumes = volumes

        self.image_paths = [p for p in all_image_paths if os.path.basename(p).split('_')[1] in selected_volumes]
        
        # Verify masks if required
        if self.load_mask:
            self.mask_paths = [p.replace("cropped_images", "cropped_masks") for p in self.image_paths]
            valid_indices = [i for i, mp in enumerate(self.mask_paths) if os.path.exists(mp)]
            self.image_paths = [self.image_paths[i] for i in valid_indices]
            self.mask_paths = [self.mask_paths[i] for i in valid_indices]
            
        # Add edge map paths if required
        if self.load_edge:
            self.edge_paths = [p.replace("cropped_images", "edge_map_images") for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = tiff.imread(img_path).astype(np.float32) / 255.0
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        result = {'image': image}

        if self.load_mask:
            mask_path = self.mask_paths[idx]
            mask = tiff.imread(mask_path).astype(np.uint8)
            if mask.shape[:2] != image.shape[:2]:
                import cv2
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            result['mask'] = mask
            
        if self.load_edge:
            edge_path = self.edge_paths[idx]
            if os.path.exists(edge_path):
                edge = tiff.imread(edge_path).astype(np.float32) / 255.0
                if edge.shape[:2] != image.shape[:2]:
                    import cv2
                    edge = cv2.resize(edge, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                result['edge'] = edge
            else:
                result['edge'] = np.zeros(image.shape[:2], dtype=np.float32)

        if self.transforms:
            # Note: A.Compose handles 'image', 'mask', 'masks' keys.
            # We treat 'edge' as an additional mask for transformation.
            if 'edge' in result:
                # Map edge to 'mask' temporarily for transform if needed, 
                # but better to use 'masks' list for multiple targets.
                transformed = self.transforms(image=result['image'], 
                                              masks=[result['mask'], result['edge']] if self.load_mask else [result['edge']])
                result['image'] = transformed['image']
                if self.load_mask:
                    result['mask'] = transformed['masks'][0]
                    result['edge'] = transformed['masks'][1]
                else:
                    result['edge'] = transformed['masks'][0]
            else:
                result = self.transforms(**result)
            
        return result

def get_train_transforms(img_size=(640, 640)):
    return A.Compose([
        A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=0),
        A.CenterCrop(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(img_size=(640, 640)):
    return A.Compose([
        A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=0),
        A.CenterCrop(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

if __name__ == "__main__":
    # Test dataset loading
    for vendor in ["Cirrus", "Spectralis", "Topcon"]:
        dataset = RETOUCHDataset(data_root="data", vendor=vendor, transforms=get_val_transforms())
        print(f"\nVendor: {vendor}, Total: {len(dataset)}")
        
        has_fluid = 0
        for i in range(len(dataset)):
            sample = dataset[i]
            if torch.any(sample['mask'] > 0):
                has_fluid += 1
        print(f"Samples with fluid: {has_fluid}/{len(dataset)}")
