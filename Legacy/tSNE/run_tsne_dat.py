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
            # The header is 227 bytes, the rest is the 3600x128 matrix
            if len(data_bytes) < 227 + 128:
                return None
            arr = np.frombuffer(data_bytes[227:], dtype=np.uint8)
            # Make sure it can be reshaped
            if len(arr) % 128 != 0:
                # Truncate to nearest multiple of 128
                arr = arr[:(len(arr)//128)*128]
            data = arr.reshape(-1, 128)
            
        max_vals = np.max(data, axis=0)
        non_zeros = np.count_nonzero(data, axis=0) + 1e-15
        density = max_vals / non_zeros
        return np.concatenate([max_vals, density])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def collect_by_date(root_folder):
    X, y = [], []
    # Classes: Lab, Field, Artificially_Generated
    for class_name in ['Lab', 'Field', 'Artificially_Generated']:
        class_dir = os.path.join(root_folder, class_name)
        if not os.path.exists(class_dir): continue
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith('.dat'):
                    feat = extract_features_dat(os.path.join(root, file))
                    if feat is not None:
                        X.append(feat)
                        y.append(class_name)
    return np.array(X), np.array(y)

def collect_by_type(root_folder):
    X, y = [], []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.dat'):
                # Determine label from path
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

def plot_2d_3d(X, y, title_prefix, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[{title_prefix}] Computing 2D TSNE...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_2d = tsne_2d.fit_transform(X)
    
    print(f"[{title_prefix}] Computing 3D TSNE...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_3d = tsne_3d.fit_transform(X)
    
    # Plot 2D
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1], 
        hue=y, palette='bright', s=50, alpha=0.7
    )
    plt.title(f'{title_prefix} - 2D t-SNE', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2d_plot.png'), dpi=300)
    plt.close()
    
    # Plot 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = list(set(y))
    colors = sns.color_palette('bright', len(unique_labels))
    label_to_color = dict(zip(unique_labels, colors))
    
    for lbl in unique_labels:
        idx = (y == lbl)
        ax.scatter(X_3d[idx, 0], X_3d[idx, 1], X_3d[idx, 2], 
                   c=[label_to_color[lbl]], label=lbl, s=30, alpha=0.7)
                   
    ax.set_title(f'{title_prefix} - 3D t-SNE', fontsize=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3d_plot.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Collecting Data/by_date...")
    X_date, y_date = collect_by_date('Data/by_date')
    if len(X_date) > 0:
        plot_2d_3d(X_date, y_date, "Data By Date", "Results/tsne_plots/data_by_date")
    
    print("Collecting Data/by_type...")
    X_type, y_type = collect_by_type('Data/by_type')
    if len(X_type) > 0:
        plot_2d_3d(X_type, y_type, "Data By Type", "Results/tsne_plots/data_by_type")
        
    print("All visualizations completed and saved.")
