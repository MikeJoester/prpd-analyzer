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

def collect_by_type(root_folder):
    X, y = [], []
    for root, _, files in os.walk(root_folder):
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
    colors = sns.color_palette('bright', len(unique_labels))
    label_to_color = dict(zip(unique_labels, colors))
    
    # Calculate global axes limits so that all plots align perfectly
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
        # Plot other points as background context
        plt.scatter(X_2d[~idx, 0], X_2d[~idx, 1], c='lightgrey', s=15, alpha=0.3, label='Other')
        # Highlight target class
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=[label_to_color[lbl]], s=50, alpha=0.9, label=lbl)
        plt.xlim(x2_min, x2_max)
        plt.ylim(y2_min, y2_max)
        plt.title(f'2D t-SNE - Class: {lbl}', fontsize=15)
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
        ax.set_title(f'3D t-SNE - Class: {lbl}', fontsize=15)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'split_3d_{lbl}.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    print("Collecting Data/by_type...")
    X_type, y_type = collect_by_type('Data/by_type')
    if len(X_type) > 0:
        plot_split(X_type, y_type, "Results/tsne_plots/data_by_type")
        
    print("Split visualizations completed and saved.")
