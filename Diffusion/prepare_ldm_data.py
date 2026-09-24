import os
import numpy as np
from PIL import Image
import shutil
import random

def process_and_save(src_path, dst_path):
    try:
        with open(src_path, 'rb') as f:
            data_bytes = f.read()
        if len(data_bytes) < 227 + 128:
            return False
        arr = np.frombuffer(data_bytes[227:], dtype=np.uint8)
        if len(arr) % 128 != 0:
            arr = arr[:(len(arr)//128)*128]
        data = arr.reshape(-1, 128)
        
        # We resize the 3600x128 matrix into a 256x256 square image for the LDM
        img = Image.fromarray(data, mode='L')
        img_resized = img.resize((256, 256), resample=Image.BILINEAR)
        img_resized.save(dst_path)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    src_root = 'Data/by_type'
    out_root = 'Data/ldm_dataset'
    
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
        
    classes = ['Corona', 'Floating', 'Particle', 'Void', 'Noise']
    
    # Create output directories
    for c in classes:
        os.makedirs(os.path.join(out_root, 'train', c), exist_ok=True)
        os.makedirs(os.path.join(out_root, 'val', c), exist_ok=True)
        
    print("Converting binary .dat files into 256x256 PNGs for Latent Diffusion...")
    
    for root, _, files in os.walk(src_root):
        for file in files:
            if file.endswith('.dat'):
                norm_root = root.replace('\\', '/')
                if 'Noise' in norm_root or 'Randomly_Generated' in norm_root:
                    lbl = 'Noise'
                elif 'Corona' in norm_root:
                    lbl = 'Corona'
                elif 'Floating' in norm_root:
                    lbl = 'Floating'
                elif 'Particle' in norm_root:
                    lbl = 'Particle'
                elif 'Void' in norm_root:
                    lbl = 'Void'
                else:
                    continue # Skip unknown
                    
                src_path = os.path.join(root, file)
                
                # 80/20 train/val split
                split = 'train' if random.random() < 0.8 else 'val'
                dst_name = file.replace('.dat', '.png')
                dst_path = os.path.join(out_root, split, lbl, dst_name)
                
                process_and_save(src_path, dst_path)
                
    # Count files
    for split in ['train', 'val']:
        total = 0
        for c in classes:
            c_dir = os.path.join(out_root, split, c)
            count = len(os.listdir(c_dir))
            total += count
            print(f"[{split}] {c}: {count} images")
        print(f"Total {split} images: {total}\n")

if __name__ == "__main__":
    main()
