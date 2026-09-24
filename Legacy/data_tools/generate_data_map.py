import os
import pandas as pd

def generate_map():
    # Dictionary to store file mapping: filename -> { ... properties ... }
    file_map = {}
    
    # 1. Parse by_date folder
    # Structure: Data/by_date/Origin/Date/Class/file.dat
    # except Artificially_Generated/unknown_date/Noise/file.dat
    by_date_dir = os.path.join('Data', 'by_date')
    for root, dirs, files in os.walk(by_date_dir):
        for f in files:
            if f.endswith('.dat'):
                # Extract path components
                rel_path = os.path.relpath(root, by_date_dir)
                parts = rel_path.split(os.sep)
                
                origin = parts[0] if len(parts) > 0 else 'Unknown'
                date = parts[1] if len(parts) > 1 else 'Unknown'
                class_name = parts[2] if len(parts) > 2 else 'Unknown'
                
                file_map[f] = {
                    'Filename': f,
                    'By_Date_Path': os.path.join(rel_path, f).replace('\\', '/'),
                    'Collection_Origin': origin,
                    'Collection_Date': date,
                    'By_Date_Class': class_name
                }
                
    # 2. Parse by_type folder
    # Structure: Data/by_type/Category/Class/file.dat
    # or Data/by_type/Category/file.dat (e.g. for Noise)
    by_type_dir = os.path.join('Data', 'by_type')
    for root, dirs, files in os.walk(by_type_dir):
        for f in files:
            if f.endswith('.dat'):
                rel_path = os.path.relpath(root, by_type_dir)
                parts = rel_path.split(os.sep)
                
                category = parts[0] if len(parts) > 0 else 'Unknown'
                
                # Some folders have subfolders for class (e.g. Lab_PD/Corona), some don't (e.g. Lab_Noise/)
                if len(parts) > 1:
                    class_name = parts[1]
                else:
                    if 'Noise' in category or 'Randomly_Generated' in category:
                        class_name = 'Noise'
                    else:
                        class_name = 'Unknown'
                
                if f in file_map:
                    file_map[f]['By_Type_Path'] = os.path.join(rel_path, f).replace('\\', '/')
                    file_map[f]['By_Type_Category'] = category
                    file_map[f]['By_Type_Class'] = class_name
                else:
                    print(f"Warning: File {f} found in by_type but not in by_date.")
                    
    # Convert to DataFrame
    df = pd.DataFrame(list(file_map.values()))
    
    # Reorder columns
    columns = [
        'Filename', 
        'Collection_Origin', 
        'Collection_Date', 
        'By_Date_Class',
        'By_Type_Category',
        'By_Type_Class',
        'By_Date_Path', 
        'By_Type_Path'
    ]
    df = df[columns]
    
    # Sort
    df = df.sort_values(by=['Collection_Origin', 'Collection_Date', 'By_Date_Class', 'Filename'])
    
    # Save to CSV
    output_file = 'Data/Data_Folders_Map.csv'
    df.to_csv(output_file, index=False)
    print(f"Successfully generated map with {len(df)} files at {output_file}")

if __name__ == "__main__":
    generate_map()
