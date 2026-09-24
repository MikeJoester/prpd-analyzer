import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Use PIL for better image handling and resizing if it's installed,
# otherwise fallback to tk.PhotoImage which supports PNG in modern Python.
try:
    import PIL
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class PlotViewerGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("t-SNE Plot Viewer")
        self.geometry("1000x800")

        self.base_dir = "Results/tsne_plots"
        
        # Categories and their corresponding folders
        self.categories = {
            "By Type": "data_by_type",
            "By Date": "data_by_date"
        }
        
        # Setup UI
        self.setup_ui()
        
        # Initial load
        self.update_image_list()

    def setup_ui(self):
        # Top Frame for controls
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Category Selection
        ttk.Label(control_frame, text="Select Category:").pack(side=tk.LEFT, padx=(0, 5))
        self.category_var = tk.StringVar(value="By Type")
        category_dropdown = ttk.Combobox(control_frame, textvariable=self.category_var, values=list(self.categories.keys()), state="readonly", width=15)
        category_dropdown.pack(side=tk.LEFT, padx=(0, 20))
        category_dropdown.bind("<<ComboboxSelected>>", self.update_image_list)

        # Image Selection
        ttk.Label(control_frame, text="Select Plot:").pack(side=tk.LEFT, padx=(0, 5))
        self.image_var = tk.StringVar()
        self.image_dropdown = ttk.Combobox(control_frame, textvariable=self.image_var, state="readonly", width=40)
        self.image_dropdown.pack(side=tk.LEFT, padx=(0, 20))
        self.image_dropdown.bind("<<ComboboxSelected>>", self.display_image)

        # Image Display Label
        self.image_label = ttk.Label(self)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def update_image_list(self, event=None):
        category = self.category_var.get()
        folder = self.categories.get(category)
        
        if not folder:
            return

        folder_path = os.path.join(self.base_dir, folder)
        
        if not os.path.exists(folder_path):
            self.image_dropdown['values'] = ["Directory not found"]
            self.image_var.set("Directory not found")
            return

        # List all png files
        images = [f for f in os.listdir(folder_path) if f.endswith(".png")]
        
        # Sort them nicely (e.g. 2d first, then 3d)
        images.sort()

        if images:
            self.image_dropdown['values'] = images
            self.image_var.set(images[0])
            self.display_image()
        else:
            self.image_dropdown['values'] = ["No images found"]
            self.image_var.set("No images found")
            self.image_label.configure(image='')

    def display_image(self, event=None):
        filename = self.image_var.get()
        category = self.category_var.get()
        folder = self.categories.get(category)
        
        if not filename or filename in ["Directory not found", "No images found"]:
            return

        filepath = os.path.join(self.base_dir, folder, filename)
        
        if not os.path.exists(filepath):
            return

        try:
            if HAS_PIL:
                # Use PIL to resize the image to fit the window while preserving aspect ratio
                img = Image.open(filepath)
                
                # Get current window size (with some padding)
                self.update_idletasks()
                max_width = self.winfo_width() - 40
                max_height = self.winfo_height() - 100
                
                if max_width <= 0 or max_height <= 0:
                    max_width, max_height = 800, 600

                # Calculate aspect ratio
                img_width, img_height = img.size
                ratio = min(max_width/img_width, max_height/img_height)
                
                # Resize if image is larger than window
                if ratio < 1:
                    new_size = (int(img_width * ratio), int(img_height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                self.tk_image = ImageTk.PhotoImage(img)
            else:
                # Fallback to standard tkinter if PIL is not available (no resizing)
                self.tk_image = tk.PhotoImage(file=filepath)
                
            self.image_label.configure(image=self.tk_image)
        except Exception as e:
            print(f"Error loading image: {e}")

if __name__ == "__main__":
    app = PlotViewerGUI()
    app.mainloop()
