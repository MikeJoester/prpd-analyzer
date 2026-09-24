import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from sklearn.manifold import TSNE
import threading

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

def collect_by_type(root_folder):
    X, y = [], []
    for parent in os.listdir(root_folder):
        parent_path = os.path.join(root_folder, parent)
        if not os.path.isdir(parent_path): continue
        for root, _, files in os.walk(parent_path):
            for file in files:
                if file.endswith('.dat'):
                    norm_root = root.replace('\\', '/')
                    if 'Noise' in norm_root or 'Randomly_Generated' in norm_root:
                        class_name = 'Noise'
                    elif 'Corona' in norm_root: class_name = 'Corona'
                    elif 'Floating' in norm_root: class_name = 'Floating'
                    elif 'Particle' in norm_root: class_name = 'Particle'
                    elif 'Void' in norm_root: class_name = 'Void'
                    else: class_name = 'Unknown'
                    
                    feat = extract_features_dat(os.path.join(root, file))
                    if feat is not None:
                        X.append(feat)
                        y.append(f"{parent}/{class_name}")
    return np.array(X), np.array(y)

def collect_by_date(root_folder):
    X, y = [], []
    for parent in ['Lab', 'Field', 'Artificially_Generated']:
        parent_path = os.path.join(root_folder, parent)
        if not os.path.exists(parent_path): continue
        for root, _, files in os.walk(parent_path):
            for file in files:
                if file.endswith('.dat'):
                    norm_root = root.replace('\\', '/')
                    if 'Noise' in norm_root or 'Artificially_Generated' in norm_root:
                        class_name = 'Noise'
                    elif 'Corona' in norm_root: class_name = 'Corona'
                    elif 'Floating' in norm_root: class_name = 'Floating'
                    elif 'Particle' in norm_root: class_name = 'Particle'
                    elif 'Void' in norm_root: class_name = 'Void'
                    else: class_name = 'Unknown'
                    
                    feat = extract_features_dat(os.path.join(root, file))
                    if feat is not None:
                        X.append(feat)
                        y.append(f"{parent}/{class_name}")
    return np.array(X), np.array(y)


class InteractiveTSNEApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interactive t-SNE Data Visualizer - Hierarchical")
        self.geometry("1400x900")
        
        self.data_cache = {
            "By Type": {"X_2d": None, "X_3d": None, "y": None},
            "By Date": {"X_2d": None, "X_3d": None, "y": None}
        }
        
        self.type_vars = {}
        self.colors_config = {}
        
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Top row
        top_ctrl = ttk.Frame(control_frame)
        top_ctrl.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Status: Ready...")
        ttk.Label(top_ctrl, textvariable=self.status_var, font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 20))
        
        self.reload_btn = ttk.Button(top_ctrl, text="Force Recompute Current Data", command=self.force_reload)
        self.reload_btn.pack(side=tk.RIGHT, padx=10)
        
        # Bottom row: Selectors
        bot_ctrl = ttk.Frame(control_frame)
        bot_ctrl.pack(side=tk.TOP, fill=tk.X)
        
        # Data Source Selection
        src_frame = ttk.LabelFrame(bot_ctrl, text="Data Source")
        src_frame.pack(side=tk.LEFT, padx=10)
        self.source_var = tk.StringVar(value="By Type")
        ttk.Radiobutton(src_frame, text="By Type (Lab_PD, Field_PD...)", variable=self.source_var, value="By Type", command=self.on_source_change).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Radiobutton(src_frame, text="By Date (Lab, Field...)", variable=self.source_var, value="By Date", command=self.on_source_change).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Dimension Selection
        dim_frame = ttk.LabelFrame(bot_ctrl, text="Graph Type")
        dim_frame.pack(side=tk.LEFT, padx=10)
        self.dim_var = tk.StringVar(value="2D")
        ttk.Radiobutton(dim_frame, text="2D", variable=self.dim_var, value="2D", command=self.update_plot).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Radiobutton(dim_frame, text="3D", variable=self.dim_var, value="3D", command=self.update_plot).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Toggle Buttons for Types (Dynamic)
        self.toggles_frame = ttk.Frame(bot_ctrl)
        self.toggles_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Plot Area
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def build_toggles(self):
        source = self.source_var.get()
        y_data = self.data_cache[source]["y"]
        
        for widget in self.toggles_frame.winfo_children():
            widget.destroy()
        self.type_vars.clear()
        
        if y_data is None:
            return
            
        unique_labels = sorted(list(set(y_data)))
        
        # Group by parent folder
        groups = {}
        for lbl in unique_labels:
            parent, child = lbl.split('/')
            if parent not in groups:
                groups[parent] = []
            groups[parent].append(child)
            
        # Assign colors based on all unique labels
        colors = sns.color_palette('hsv', len(unique_labels))
        self.colors_config = dict(zip(unique_labels, colors))
        
        for parent, children in groups.items():
            frame = ttk.LabelFrame(self.toggles_frame, text=parent)
            frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
            
            # Select/Deselect All for the group
            def toggle_group(p=parent, c=children):
                # check state of first one to flip
                state = not self.type_vars[f"{p}/{c[0]}"].get()
                for child in c:
                    self.type_vars[f"{p}/{child}"].set(state)
                self.update_plot()
                
            btn = ttk.Button(frame, text="Toggle All", command=toggle_group, width=10)
            btn.pack(side=tk.TOP, pady=2)
            
            for child in children:
                full_label = f"{parent}/{child}"
                var = tk.BooleanVar(value=True)
                self.type_vars[full_label] = var
                chk = ttk.Checkbutton(frame, text=child, variable=var, command=self.update_plot)
                chk.pack(side=tk.TOP, anchor='w', padx=5, pady=2)

    def on_source_change(self):
        self.load_data()
        
    def force_reload(self):
        self.load_data(force=True)

    def load_data(self, force=False):
        source = self.source_var.get()
        
        if not force and self.data_cache[source]["X_2d"] is not None:
            self.build_toggles()
            self.update_plot()
            return
            
        self.status_var.set(f"Status: Loading {source} Data and Computing t-SNE... Please wait.")
        self.update_idletasks()
        
        self.reload_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._process_data_thread, args=(source, force), daemon=True).start()

    def _process_data_thread(self, source, force):
        cache_file = "artifacts/tsne_cache/tsne_gui_cache_type_v2.npz" if source == "By Type" else "artifacts/tsne_cache/tsne_gui_cache_date_v2.npz"
        
        if not force and os.path.exists(cache_file):
            self.status_var.set(f"Status: Loading {source} from disk cache...")
            data = np.load(cache_file, allow_pickle=True)
            self.data_cache[source]["X_2d"] = data['X_2d']
            self.data_cache[source]["X_3d"] = data['X_3d']
            self.data_cache[source]["y"] = data['y']
        else:
            if source == "By Type":
                self.status_var.set("Status: Reading .dat files from Data/by_type... (~10s)")
                X, y = collect_by_type('Data/by_type')
            else:
                self.status_var.set("Status: Reading .dat files from Data/by_date... (~10s)")
                X, y = collect_by_date('Data/by_date')
            
            if len(X) == 0:
                self.status_var.set(f"Status: Error - No data found for {source}!")
                self.after(0, lambda: self.reload_btn.config(state=tk.NORMAL))
                return
                
            self.status_var.set(f"Status: Computing 2D t-SNE for {source}... (~15s)")
            tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
            X_2d = tsne_2d.fit_transform(X)
            
            self.status_var.set(f"Status: Computing 3D t-SNE for {source}... (~15s)")
            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
            X_3d = tsne_3d.fit_transform(X)
            
            self.data_cache[source]["X_2d"] = X_2d
            self.data_cache[source]["X_3d"] = X_3d
            self.data_cache[source]["y"] = y
            
            np.savez(cache_file, X_2d=X_2d, X_3d=X_3d, y=y)
            
        self.status_var.set(f"Status: Done! Loaded {len(self.data_cache[source]['y'])} samples.")
        
        self.after(0, lambda: self.reload_btn.config(state=tk.NORMAL))
        self.after(0, self.build_toggles)
        self.after(0, self.update_plot)

    def update_plot(self):
        source = self.source_var.get()
        X_2d = self.data_cache[source]["X_2d"]
        X_3d = self.data_cache[source]["X_3d"]
        y = self.data_cache[source]["y"]
        
        if X_2d is None or X_3d is None or len(self.type_vars) == 0:
            return
            
        self.fig.clf()
        
        is_3d = (self.dim_var.get() == "3D")
        
        if is_3d:
            ax = self.fig.add_subplot(111, projection='3d')
        else:
            ax = self.fig.add_subplot(111)
            
        has_data = False
        
        for full_label, var in self.type_vars.items():
            if var.get():  # If toggled ON
                idx = (y == full_label)
                if np.any(idx):
                    has_data = True
                    color = [self.colors_config[full_label]]
                    # Replace / with - for legend
                    display_label = full_label.replace('/', ' - ')
                    if is_3d:
                        ax.scatter(X_3d[idx, 0], X_3d[idx, 1], X_3d[idx, 2], 
                                   c=color, label=display_label, s=30, alpha=0.7)
                    else:
                        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], 
                                   c=color, label=display_label, s=50, alpha=0.7)
                                   
        if has_data:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_title(f"Interactive t-SNE - {source} ({self.dim_var.get()})", fontsize=15)
        else:
            ax.set_title("No Data Types Selected", fontsize=15)
            
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = InteractiveTSNEApp()
    app.mainloop()
