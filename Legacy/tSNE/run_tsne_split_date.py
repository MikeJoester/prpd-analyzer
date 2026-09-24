import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def extract_features_dat(file_path):
    try:
        with open(file_path, 'rb') as f:
            data_bytes = f.read()
            if len(data_bytes) < 227 + 128:
                return None
            arr = np.frombuffer(data_bytes[227:], dtype=np.uint8)
            if len(arr) % 128 != 0:
                arr = arr[:(len(arr)//128)*128]
            data = arr.reshape(-1, 128)
            
        max_vals = np.max(data, axis=0)
        non_zeros = np.count_nonzero(data, axis=0) + 1e-15
        density = max_vals / non_zeros
        return np.concatenate([max_vals, density])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def build_label_map(type_folder):
    """Scan Data/by_type to map filenames to their PD class."""
    label_map = {}
    for root, _, files in os.walk(type_folder):
        for file in files:
            if file.endswith('.dat'):
                norm_root = root.replace('\\', '/')
                if 'Noise' in norm_root or 'Randomly_Generated' in norm_root:
                    label = 'Noise'
                elif 'Corona' in norm_root:
                    label = 'Corona'
                elif 'Floating' in norm_root:
                    label = 'Floating'
                elif 'Particle' in norm_root:
                    label = 'Particle'
                elif 'Void' in norm_root:
                    label = 'Void'
                else:
                    label = 'Unknown'
                label_map[file] = label
    return label_map

def collect_by_date_with_pd_labels(root_folder, label_map):
    X, y = [], []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.dat'):
                label = label_map.get(file, 'Unknown')
                feat = extract_features_dat(os.path.join(root, file))
                if feat is not None:
                    X.append(feat)
                    y.append(label)
    return np.array(X), np.array(y)

def plot_split(X, y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    print("Computing 2D TSNE...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_2d = tsne_2d.fit_transform(X)
    
    print("Computing 3D TSNE...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_3d = tsne_3d.fit_transform(X)
    
    unique_labels = list(set(y))
    if 'Unknown' in unique_labels:
        unique_labels.remove('Unknown')
        
    colors = sns.color_palette('bright', len(unique_labels))
    label_to_color = dict(zip(unique_labels, colors))
    if 'Unknown' in set(y):
        label_to_color['Unknown'] = (0.5, 0.5, 0.5)
    
    pad2_x = (X_2d[:,0].max() - X_2d[:,0].min()) * 0.05
    pad2_y = (X_2d[:,1].max() - X_2d[:,1].min()) * 0.05
    x2_min, x2_max = X_2d[:,0].min()-pad2_x, X_2d[:,0].max()+pad2_x
    y2_min, y2_max = X_2d[:,1].min()-pad2_y, X_2d[:,1].max()+pad2_y
    
    pad3_x = (X_3d[:,0].max() - X_3d[:,0].min()) * 0.05
    pad3_y = (X_3d[:,1].max() - X_3d[:,1].min()) * 0.05
    pad3_z = (X_3d[:,2].max() - X_3d[:,2].min()) * 0.05
    x3_min, x3_max = X_3d[:,0].min()-pad3_x, X_3d[:,0].max()+pad3_x
    y3_min, y3_max = X_3d[:,1].min()-pad3_y, X_3d[:,1].max()+pad3_y
    z3_min, z3_max = X_3d[:,2].min()-pad3_z, X_3d[:,2].max()+pad3_z

    for lbl in unique_labels:
        print(f"Generating split plots for {lbl}...")
        idx = (y == lbl)
        
        # 2D plot
        plt.figure(figsize=(10, 8))
        plt.scatter(X_2d[~idx, 0], X_2d[~idx, 1], c='lightgrey', s=15, alpha=0.3, label='Other')
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=[label_to_color[lbl]], s=50, alpha=0.9, label=lbl)
        plt.xlim(x2_min, x2_max)
        plt.ylim(y2_min, y2_max)
        plt.title(f'2D t-SNE (by_date) - Class: {lbl}', fontsize=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'split_2d_{lbl}.png'), dpi=300)
        plt.close()
        
        # 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_3d[~idx, 0], X_3d[~idx, 1], X_3d[~idx, 2], c='lightgrey', s=10, alpha=0.15, label='Other')
        ax.scatter(X_3d[idx, 0], X_3d[idx, 1], X_3d[idx, 2], c=[label_to_color[lbl]], s=30, alpha=0.9, label=lbl)
        ax.set_xlim(x3_min, x3_max)
        ax.set_ylim(y3_min, y3_max)
        ax.set_zlim(z3_min, z3_max)
        ax.set_title(f'3D t-SNE (by_date) - Class: {lbl}', fontsize=15)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'split_3d_{lbl}.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    print("Building filename -> label map from Data/by_type...")
    label_map = build_label_map('Data/by_type')
    
    print("Collecting Data/by_date...")
    X_date, y_date = collect_by_date_with_pd_labels('Data/by_date', label_map)
    
    if len(X_date) > 0:
        plot_split(X_date, y_date, "Results/tsne_plots/data_by_date")
        
    print("Split visualizations for by_date completed and saved.")
