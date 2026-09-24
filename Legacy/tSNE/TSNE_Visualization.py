import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def get_standard_label(folder_name):
    """Maps messy folder names to the 6 required categories."""
    fn = folder_name.lower()
    
    if 'normal' in fn: return 'Noise'
    if 'corona' in fn: return 'Corona'
    if 'turn_to_turn' in fn or 'turn to turn' in fn: return 'Turn_to_Turn'
    if 'floating' in fn: return 'Floating'
    if 'surface' in fn: return 'Surface'
    if 'void' in fn: return 'Void'
    
    return None # Skip if it doesn't match our categories

def extract_features(file_path):
    """Extracts the (1, 256) SVM feature vector."""
    try:
        data = np.loadtxt(file_path, delimiter=",")
        # Feature 1: Max (128,)
        max_vals = np.max(data, axis=0)
        # Feature 2: Density (128,)
        non_zeros = np.count_nonzero(data, axis=0) + 1e-15
        density = max_vals / non_zeros
        return np.concatenate([max_vals, density])
    except Exception:
        return None

def collect_standardized_data(root_folders):
    X, y = [], []
    
    for root in root_folders:
        if not os.path.exists(root): continue
        
        for folder_name in os.listdir(root):
            label = get_standard_label(folder_name)
            if label is None: continue # Ignore folders that don't match criteria
            
            folder_path = os.path.join(root, folder_name)
            if os.path.isdir(folder_path):
                print(f"Processing: {label} (from {folder_name})")
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        feat = extract_features(os.path.join(folder_path, file))
                        if feat is not None:
                            X.append(feat)
                            y.append(label)
                            
    return np.array(X), np.array(y)

def run_visualization(X, y):
    print(f"\nTotal Samples: {len(X)}")
    print("Computing t-SNE (this may take a minute)...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1], 
        hue=y, palette='bright', s=50, alpha=0.7,
        hue_order=['Noise', 'Corona', 'Turn_to_Turn', 'Floating', 'Surface', 'Void']
    )
    
    plt.title('t-SNE Visualization: Combined Partial Discharge Datasets', fontsize=15)
    plt.legend(title='Discharge Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folders = ["Data/0820", "Data/0903", "Data/250915_to_250916_MTR_NEW_DB_KERI"]
    X_data, y_labels = collect_standardized_data(folders)
    
    if len(X_data) > 0:
        run_visualization(X_data, y_labels)
    else:
        print("Error: No data found. Verify folder paths.")