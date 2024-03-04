import math
import rasterio
from rasterio.windows import Window

in_folder_path = "./orthomosaic/"
out_folder_path = "./rasterized/"
mosaic_name = "DJI_0015.JPG"
img_path = in_folder_path + mosaic_name
n = 20

def chop2nWindows(image_path, n):
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        
        n_split_width = int(math.ceil(math.sqrt(n * width / height)))
        n_split_height = int(math.ceil(n / n_split_width))

        window_width = math.floor(width / n_split_width)
        window_height = math.floor(height / n_split_height)

        windows = []
        for row in range(n_split_height):
            for col in range(n_split_width):
                w_col = col * window_width
                w_row = row * window_height

                if col == n_split_width - 1:
                    window_width_adj = width - w_col
                else:
                    window_width_adj = window_width
                
                if row == n_split_height - 1:
                    window_height_adj = height - w_row
                else:
                    window_height_adj = window_height

                window = Window(w_col, w_row, window_width_adj, window_height_adj)
                transform = rasterio.windows.transform(window, src.transform)
                windows.append((window, transform))

        return windows
    
def saveWindow(image_path, window, transform, output_path):
    with rasterio.open(image_path) as src:
        window_data = src.read(window=window)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(window_data)
        print(f"Saved window to {output_path}")
                
windows = chop2nWindows(img_path, n)
print(f"Generated {len(windows)} windows.")
for i, (window, transform) in enumerate(windows):
    print(window)
    output_path = out_folder_path + f"window_{i}.tif"
    saveWindow(img_path, window, transform, output_path)
