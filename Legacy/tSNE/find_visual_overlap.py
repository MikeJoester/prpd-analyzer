import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

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
        return None

def build_label_map(type_folder):
    label_map = {}
    for root, _, files in os.walk(type_folder):
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
                    lbl = 'Unknown'
                label_map[file] = lbl
    return label_map

def main():
    print("Building filename -> label map from Data/by_type...")
    label_map = build_label_map('Data/by_type')
    
    X = []
    y = []
    file_paths = []
    
    print("Scanning Data/by_date...")
    for root, _, files in os.walk('Data/by_date'):
        for file in files:
            if file.endswith('.dat'):
                file_path = os.path.join(root, file)
                feat = extract_features_dat(file_path)
                if feat is not None:
                    label = label_map.get(file, 'Unknown')
                    X.append(feat)
                    y.append(label)
                    file_paths.append(file_path)
                    
    X = np.array(X)
    y = np.array(y)
    file_paths = np.array(file_paths)
    
    print("Computing 2D TSNE...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_2d = tsne_2d.fit_transform(X)
    
    print("Finding visual overlaps using k-Nearest Neighbors in 2D space...")
    # Define macro classes
    is_noise = (y == 'Noise')
    is_pd = (y != 'Noise') & (y != 'Unknown')
    macro_y = np.zeros(len(y))
    macro_y[is_noise] = 0
    macro_y[is_pd] = 1
    macro_y[(y == 'Unknown')] = -1
    
    # We will look at the 15 nearest neighbors for each point
    k = 15
    nn = NearestNeighbors(n_neighbors=k+1) # +1 because the point itself is the first neighbor
    nn.fit(X_2d)
    distances, indices = nn.kneighbors(X_2d)
    
    overlap_indices = []
    report_lines = ["# Visual Overlap Report (Noise vs PD)\n\n"]
    report_lines.append(f"This report lists files that visually overlap in the 2D t-SNE space. An overlap is defined as a point whose {k} nearest neighbors contain a significant mixture of both Noise and PD samples.\n\n")
    
    for i in range(len(X_2d)):
        if macro_y[i] == -1: continue # Skip unknown
        
        neighbor_macros = macro_y[indices[i][1:]] # exclude self
        noise_count = np.sum(neighbor_macros == 0)
        pd_count = np.sum(neighbor_macros == 1)
        
        # Criteria: If a Noise point is surrounded by at least 3 PD points, 
        # or a PD point is surrounded by at least 3 Noise points, flag it.
        is_overlap = False
        if macro_y[i] == 0 and pd_count >= 3:
            is_overlap = True
        elif macro_y[i] == 1 and noise_count >= 3:
            is_overlap = True
            
        if is_overlap:
            overlap_indices.append(i)
            report_lines.append(f"- **{file_paths[i]}** (Actual Label: {y[i]}) -> Neighbors in 2D: {noise_count} Noise, {pd_count} PD\n")
            
    print(f"Found {len(overlap_indices)} visually overlapping points.")
    
    # Save the report
    with open('Results/reports/visual_overlap_report.md', 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
        
    print("Generating plot...")
    plt.figure(figsize=(12, 10))
    
    unique_labels = list(set(y))
    if 'Unknown' in unique_labels: unique_labels.remove('Unknown')
    colors = sns.color_palette('bright', len(unique_labels))
    label_to_color = dict(zip(unique_labels, colors))
    if 'Unknown' in set(y): label_to_color['Unknown'] = (0.5, 0.5, 0.5)
    
    for lbl in unique_labels:
        idx = (y == lbl)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=[label_to_color[lbl]], s=40, alpha=0.7, label=lbl)
        
    if overlap_indices:
        cx = X_2d[overlap_indices, 0]
        cy = X_2d[overlap_indices, 1]
        plt.scatter(cx, cy, facecolors='none', edgecolors='red', s=150, linewidth=2.0, label='Boundary Overlaps (Noise vs PD)')
        
    plt.title('2D t-SNE with Highlighted Boundary Overlaps (Noise vs PD)', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_dir = 'Results/tsne_plots/data_by_date'
    os.makedirs(save_dir, exist_ok=True)
    out_img = os.path.join(save_dir, '2d_plot_visual_overlaps.png')
    plt.savefig(out_img, dpi=300)
    plt.close()
    
    print(f"Process complete! Plot saved to {out_img}.")

if __name__ == '__main__':
    main()
