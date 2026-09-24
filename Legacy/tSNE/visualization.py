import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def combine_csv_files_in_folder(folder_path):
    """
    Load and combine all CSV files in a folder
    
    Args:
        folder_path: Path to folder containing CSV files
        
    Returns:
        combined_data: Combined 2D numpy array or None if no files found
    """
    csv_files = sorted(folder_path.glob("*.csv"))
    
    if not csv_files:
        return None
    
    all_data = []
    for csv_file in csv_files:
        try:
            data = np.loadtxt(fname=csv_file, delimiter=",")
            if data.ndim == 1:  # Single row
                data = data.reshape(1, -1)
            all_data.append(data)
        except Exception as e:
            print(f"    Warning: Could not load {csv_file.name}: {e}")
    
    if not all_data:
        return None
    
    # Combine all data
    combined_data = np.vstack(all_data)
    return combined_data

def visualize_combined_pd_data(label_name, combined_data, output_dir):
    """
    Create 2D phase-amplitude scatter plot for combined PD data
    
    Args:
        label_name: Name of the label/class
        combined_data: Combined 2D numpy array from multiple CSV files
        output_dir: Directory to save visualizations
    """
    try:
        print(f"  Generating visualization for: {label_name}")
        
        # ===== 2D Phase-Amplitude Scatter Plot =====
        # Use all data for phase-amplitude plot
        d = combined_data.flatten()
        
        # Create phase angles - map columns to phases (0-360°)
        n_phases = combined_data.shape[1]
        radio = np.linspace(0, 360, n_phases)
        rows = np.array([radio for _ in range(combined_data.shape[0])])
        
        df = pd.DataFrame({"rows": rows.flatten(), "column": d})
        
        # Filter out zeros BEFORE creating the map column
        df = df[df["column"] != 0].reset_index(drop=True)
        
        if len(df) == 0:
            print(f"    ✗ No non-zero data for {label_name}")
            return
        
        # Now create map and count
        df["map"] = df.apply(lambda x: f"{x['column']:.2f}, {x['rows']:.2f}", axis=1)
        count_df = df["map"].value_counts().reset_index()
        count_df.columns = ["map", "count"]  # Ensure correct column names
        
        # Merge to get counts for each point
        df = pd.merge(df, count_df, left_on="map", right_on="map")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = plt.scatter(df["rows"], df["column"], s=1.55, 
                             c=df["count"]*100, cmap='cool', alpha=0.7, 
                             vmin=0, vmax=combined_data.shape[0])
        
        plt.xticks(np.arange(0, 360+1, 60))
        plt.yticks(np.arange(0, 255+1, 50))
        plt.xlabel("Phase (°)")
        plt.ylabel("Amplitude of PDs")
        plt.title(f"2D Phase-Amplitude - {label_name}")
        
        cbar = plt.colorbar(scatter)
        cbar.set_label("Number of PDs")
        
        result_2d = output_dir / f"{label_name}_2D_PhaseAmplitude.png"
        plt.savefig(result_2d, dpi=350, format='png', bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved 2D: {result_2d}")
        
    except Exception as e:
        print(f"    ✗ Error generating visualization for {label_name}: {e}")
        import traceback
        traceback.print_exc()

def process_all_datasets():
    """
    Process all datasets (0820, 0903, 250915_to_250916_MTR_NEW_DB_KERI)
    and generate visualizations for each label
    """
    data_root = Path("Data")
    output_root = Path("../report_1_results/Data_visualizations")
    
    if not data_root.exists():
        print(f"Error: Data folder not found at {data_root.absolute()}")
        return
    
    # Create output directory if it doesn't exist
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_root.absolute()}\n")
    
    # Get all dataset folders (0820, 0903, 250915_to_250916_MTR_NEW_DB_KERI)
    dataset_folders = [d for d in data_root.iterdir() if d.is_dir()]
    
    if not dataset_folders:
        print(f"No dataset folders found in {data_root}")
        return
    
    print(f"Found {len(dataset_folders)} dataset(s)\n")
    
    total_labels = 0
    
    for dataset_folder in sorted(dataset_folders):
        dataset_name = dataset_folder.name
        print(f"Processing dataset: {dataset_name}")
        
        # Get all label folders within this dataset
        label_folders = [d for d in dataset_folder.iterdir() if d.is_dir()]
        
        if not label_folders:
            print(f"  No label folders found in {dataset_name}\n")
            continue
        
        print(f"  Found {len(label_folders)} label(s)\n")
        
        for label_folder in sorted(label_folders):
            label_name = label_folder.name
            
            # Check if there are CSV files in this label folder
            csv_files = list(label_folder.glob("*.csv"))
            
            if not csv_files:
                print(f"  Skipping {label_name} (no CSV files)")
                continue
            
            print(f"  [{total_labels + 1}] Processing: {label_name}")
            print(f"      Found {len(csv_files)} CSV files")
            
            # Combine all CSV files in this label folder
            combined_data = combine_csv_files_in_folder(label_folder)
            
            if combined_data is None:
                print(f"      ✗ Failed to load data")
                continue
            
            print(f"      Combined data shape: {combined_data.shape}")
            
            # Create visualization
            visualize_combined_pd_data(label_name, combined_data, output_root)
            total_labels += 1
            print()
        
        print()
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Processed {total_labels} labels")
    print(f"✓ All 2D visualizations saved to: {output_root.absolute()}")

if __name__ == "__main__":
    process_all_datasets()