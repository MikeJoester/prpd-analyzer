import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import hashlib

def extract_features_and_hash(file_path):
    try:
        with open(file_path, 'rb') as f:
            data_bytes = f.read()
            if len(data_bytes) < 227 + 128:
                return None, None
            # Skip the 227-byte header and read the PRPD matrix
            arr = np.frombuffer(data_bytes[227:], dtype=np.uint8)
            if len(arr) % 128 != 0:
                arr = arr[:(len(arr)//128)*128]
            data = arr.reshape(-1, 128)
            
        # Hash the exact binary PRPD matrix
        data_hash = hashlib.md5(arr.tobytes()).hexdigest()
            
        # Compute features for t-SNE
        max_vals = np.max(data, axis=0)
        non_zeros = np.count_nonzero(data, axis=0) + 1e-15
        density = max_vals / non_zeros
        features = np.concatenate([max_vals, density])
        
        return features, data_hash
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def build_label_map(type_folder):
    """
    Scans Data/by_type. If a filename appears in multiple folders, 
    it collects all labels to detect contradictions.
    """
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
                    
                if file not in label_map:
                    label_map[file] = set()
                label_map[file].add(lbl)
    return label_map

def main():
    print("Building filename -> label map from Data/by_type...")
    label_map = build_label_map('Data/by_type')
    
    print("Scanning Data/by_date to find overlapping data matrices...")
    hash_to_files = {}
    
    X = []
    y = []
    file_paths = []
    
    for root, _, files in os.walk('Data/by_date'):
        for file in files:
            if file.endswith('.dat'):
                file_path = os.path.join(root, file)
                feat, h = extract_features_and_hash(file_path)
                if feat is not None:
                    # Get labels assigned to this filename
                    lbls = label_map.get(file, {'Unknown'})
                    main_lbl = list(lbls)[0]
                    
                    X.append(feat)
                    y.append(main_lbl)
                    file_paths.append(file_path)
                    
                    if h not in hash_to_files:
                        hash_to_files[h] = []
                    hash_to_files[h].append((file_path, lbls))
    
    X = np.array(X)
    y = np.array(y)
    file_paths = np.array(file_paths)
    
    print("Identifying category contradictions...")
    contradiction_indices = []
    report_lines = ["# Overlapping Data (Category Contradictions) Report\n\n"]
    report_lines.append("This report lists files that contain the **exact same PRPD data matrix**, but have been assigned conflicting labels (e.g., both Noise and PD).\n\n")
    
    contradiction_count = 0
    for h, files_info in hash_to_files.items():
        all_labels = set()
        for fp, lbls in files_info:
            all_labels.update(lbls)
            
        # A contradiction occurs if the same underlying data matrix is tied to multiple distinct categories
        if len(all_labels) > 1:
            contradiction_count += 1
            report_lines.append(f"### Contradiction #{contradiction_count} (Hash: {h})\n")
            report_lines.append(f"**Conflicting Labels:** {', '.join(all_labels)}\n")
            for fp, lbls in files_info:
                report_lines.append(f"- File: `{fp}` (Assigned: {', '.join(lbls)})\n")
                
                # Find index to circle in plot
                idx = np.where(file_paths == fp)[0][0]
                contradiction_indices.append(idx)
            report_lines.append("\n")
            
    # Save the report as an artifact
    report_path = os.path.join(os.environ.get('APPDATA', ''), '..', 'Local', 'Temp', 'overlapping_data_report.md')
    if not os.path.exists('overlapping_data_report.md'):
        pass
    with open('Results/reports/overlapping_data_report.md', 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
        
    print(f"Found {contradiction_count} distinct data matrices involved in category contradictions.")
    print("Report saved to overlapping_data_report.md.")
    
    print("Computing 2D TSNE for visualization...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_2d = tsne_2d.fit_transform(X)
    
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
        
    if contradiction_indices:
        cx = X_2d[contradiction_indices, 0]
        cy = X_2d[contradiction_indices, 1]
        plt.scatter(cx, cy, facecolors='none', edgecolors='red', s=200, linewidth=2.5, label='Overlapping (Contradictory)')
        
    plt.title('2D t-SNE with Highlighted Category Contradictions', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_dir = 'Results/tsne_plots/data_by_date'
    os.makedirs(save_dir, exist_ok=True)
    out_img = os.path.join(save_dir, '2d_plot_overlaps.png')
    plt.savefig(out_img, dpi=300)
    plt.close()
    
    print(f"Process complete! Plot saved to {out_img}.")

if __name__ == '__main__':
    main()
