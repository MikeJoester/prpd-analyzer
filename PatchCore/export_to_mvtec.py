import os
import sys
import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
from EfficientAD.train_efficient_ad import PRPDAnomalyDataset

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def export_mvtec():
    ai_data_root = os.path.join(REPO_ROOT, "artifacts", "ai_data_20260911_011704")
    dest_dir = os.path.join(REPO_ROOT, "PatchCore", "patchcore_data", "prpd")
    
    train_good_dir = os.path.join(dest_dir, "train", "good")
    test_good_dir = os.path.join(dest_dir, "test", "good")
    test_defect_dir = os.path.join(dest_dir, "test", "defective")
    
    create_dir(train_good_dir)
    create_dir(test_good_dir)
    create_dir(test_defect_dir)

    print("Loading datasets to extract exact train/test splits...")
    
    # 1. Train Good (Lab PD + 70% Field PD)
    train_dataset = PRPDAnomalyDataset(ai_data_root, mode='train')
    for idx in range(len(train_dataset)):
        tensor = train_dataset.data[idx] # shape (128, 3600)
        crop = tensor[:, :256]
        if crop.max() > 0:
            crop = (crop / crop.max()) * 255.0
        crop = crop.astype(np.uint8)
        rgb_img = np.stack((crop,)*3, axis=-1)
        img = Image.fromarray(rgb_img)
        img.save(os.path.join(train_good_dir, f"{idx:04d}.png"))
    print(f"Exported {len(train_dataset)} Train Good images.")

    # 2. Test Good (30% Unseen Field PD)
    test_dataset = PRPDAnomalyDataset(ai_data_root, mode='test')
    for idx in range(len(test_dataset)):
        tensor = test_dataset.data[idx]
        crop = tensor[:, :256]
        if crop.max() > 0:
            crop = (crop / crop.max()) * 255.0
        crop = crop.astype(np.uint8)
        rgb_img = np.stack((crop,)*3, axis=-1)
        img = Image.fromarray(rgb_img)
        img.save(os.path.join(test_good_dir, f"{idx:04d}.png"))
    print(f"Exported {len(test_dataset)} Test Good images.")

    # 3. Test Defective (Field Noise) with Dummy Ground Truth Masks
    noise_path = os.path.join(ai_data_root, "raw_tensors_v1", "Field Noise.npz")
    noise_data = np.load(noise_path)['raw']
    gt_defect_dir = os.path.join(dest_dir, "ground_truth", "defective")
    create_dir(gt_defect_dir)
    
    for idx in range(len(noise_data)):
        tensor = noise_data[idx]
        crop = tensor[:, :256]
        if crop.max() > 0:
            crop = (crop / crop.max()) * 255.0
        crop = crop.astype(np.uint8)
        rgb_img = np.stack((crop,)*3, axis=-1)
        img = Image.fromarray(rgb_img)
        filename = f"{idx:04d}_mask.png" # Standard MVTec mask naming convention
        
        # Save defective image
        img.save(os.path.join(test_defect_dir, f"{idx:04d}.png"))
        
        # Create and save dummy mask (all 255s)
        mask_img = np.full((128, 256), 255, dtype=np.uint8)
        mask = Image.fromarray(mask_img)
        mask.save(os.path.join(gt_defect_dir, filename))
        
    print(f"Exported {len(noise_data)} Test Defective images and dummy masks.")

    print("MVTec export complete!")

if __name__ == "__main__":
    export_mvtec()
