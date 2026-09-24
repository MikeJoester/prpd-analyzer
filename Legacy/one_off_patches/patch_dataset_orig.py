import re

with open("src/prpd_anomaly/train_efficient_ad.py", "r") as f:
    content = f.read()

new_dataset_class = """class PRPDAnomalyDataset(Dataset):
    def __init__(self, ai_data_root, mode='train'):
        self.ai_data_root = Path(ai_data_root)
        self.mode = mode
        
        # Load Lab PD
        lab_pd_path = self.ai_data_root / "raw_tensors_v1" / "Lab PD.npz"
        if lab_pd_path.exists():
            lab_pd = np.load(lab_pd_path)['raw']
        else:
            lab_pd = np.zeros((0, 128, 3600), dtype=np.uint8)
            
        # Load Field PD
        field_pd_path = self.ai_data_root / "raw_tensors_v1" / "Field PD.npz"
        if field_pd_path.exists():
            field_pd = np.load(field_pd_path)['raw']
        else:
            field_pd = np.zeros((0, 128, 3600), dtype=np.uint8)
            
        # Deterministic split
        np.random.seed(42)
        
        # Split Field PD (70/15/15)
        field_indices = np.random.permutation(len(field_pd))
        field_train_end = int(0.7 * len(field_pd))
        field_val_end = int(0.85 * len(field_pd))
        
        field_train = field_pd[field_indices[:field_train_end]]
        field_val = field_pd[field_indices[field_train_end:field_val_end]]
        field_test = field_pd[field_indices[field_val_end:]]
        
        if mode == 'train':
            # 100% Lab PD + 70% Field PD
            self.data = np.concatenate([lab_pd, field_train], axis=0)
        elif mode == 'val':
            # 15% Field PD
            self.data = field_val
        else: # test
            # 15% Field PD
            self.data = field_test
            
        print(f"Loaded {len(self.data)} samples for {mode} mode.")

    def __len__(self):
"""

content = re.sub(
    r"class PRPDAnomalyDataset\(Dataset\):.*?def __len__\(self\):",
    new_dataset_class,
    content,
    flags=re.DOTALL
)

with open("src/prpd_anomaly/train_efficient_ad.py", "w") as f:
    f.write(content)
print("Updated Dataset back to original specs.")
