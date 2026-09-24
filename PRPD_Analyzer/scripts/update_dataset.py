import csv
import os
import shutil
from pathlib import Path

def update_dataset(csv_path, data_dir):
    csv_path = Path(csv_path)
    data_dir = Path(data_dir)
    
    label_changes = []
    excluded = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
        mode = None
        for row in reader:
            if not row or not any(row):
                continue
                
            if row[0] == 'Environment' and 'Previous Classification' in row:
                mode = 'label_changes'
                continue
            elif row[0] == 'Environment' and 'Current Classification' in row:
                mode = 'excluded'
                continue
                
            if mode == 'label_changes' and row[0] in ['Field', 'Lab']:
                # Environment, Measurement Date, Previous Classification, New Classification, File Name
                label_changes.append({
                    'env': row[0],
                    'date': row[1],
                    'prev_class': row[2],
                    'new_class': row[3],
                    'filename': row[4]
                })
            elif mode == 'excluded' and row[0] in ['Field', 'Lab']:
                # Environment, Measurement Date, Current Classification, File Name, ...
                excluded.append({
                    'env': row[0],
                    'date': row[1],
                    'curr_class': row[2],
                    'filename': row[3]
                })

    print(f"Found {len(label_changes)} files to change labels.")
    print(f"Found {len(excluded)} files to exclude (delete).")

    # Build a lookup map of filename -> list of paths
    print("Scanning Data directory for files...")
    all_files = list(data_dir.rglob("*.dat"))
    file_map = {}
    for p in all_files:
        file_map.setdefault(p.name, []).append(p)
        
    # Process Label Changes
    moved_count = 0
    for change in label_changes:
        fname = change['filename']
        prev_cls = change['prev_class']
        new_cls = change['new_class']
        
        paths = file_map.get(fname, [])
        if not paths:
            print(f"[Warning] Label Change: File not found: {fname}")
            continue
            
        for path in paths:
            # Check if the previous class is in the path's parts
            if prev_cls in path.parts:
                # Construct new path by replacing the class folder
                new_parts = list(path.parts)
                # We replace the last occurrence of prev_cls (closest to filename)
                idx = len(new_parts) - 1 - new_parts[::-1].index(prev_cls)
                new_parts[idx] = new_cls
                new_path = Path(*new_parts)
                
                # Create parent directories if they don't exist
                new_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move
                shutil.move(str(path), str(new_path))
                print(f"Moved: {path.relative_to(data_dir)} -> {new_path.relative_to(data_dir)}")
                moved_count += 1
            else:
                print(f"[Warning] File {fname} found at {path}, but '{prev_cls}' not in path.")

    # Process Exclusions
    deleted_count = 0
    for excl in excluded:
        fname = excl['filename']
        paths = file_map.get(fname, [])
        if not paths:
            print(f"[Warning] Exclude: File not found: {fname}")
            continue
            
        for path in paths:
            if path.exists():
                path.unlink()
                print(f"Deleted: {path.relative_to(data_dir)}")
                deleted_count += 1

    print(f"\nUpdate Complete. Moved {moved_count} files, Deleted {deleted_count} files.")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    csv_file = repo_root / "Data" / "Dataset_Change_Log_260909.csv"
    data_folder = repo_root / "PRPD_Analyzer" / "Data"
    update_dataset(csv_file, data_folder)
