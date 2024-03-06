import math
import rasterio
from rasterio.windows import Window
from rasterio.env import Env

in_folder_path = "./orthomosaic/"
out_folder_path = "./rasterized/"
mosaic_name = "pumpkin_patch_cut.jpg"
img_path = in_folder_path + mosaic_name
n = 9

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

                if col == n_split_width - 1: #Last column
                    window_width_adj = width - w_col
                else:
                    window_width_adj = window_width
                
                if row == n_split_height - 1: #Last row
                    window_height_adj = height - w_row
                else:
                    window_height_adj = window_height

                window = Window(w_col, w_row, window_width_adj, window_height_adj)
                #transform = rasterio.windows.transform(window, src.transform)
                windows.append(window)

        return windows
    
def saveWindow(image_path, window, output_path):
    with Env(GDAL_PAM_ENABLED='NO'):    #Disable the creation of an .aux.xml file for each .jpg file
        with rasterio.open(image_path) as src:
            window_data = src.read(window=window)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "JPEG",
                "height": window.height,
                "width": window.width,
                "dtype": 'uint8',
                "transform": rasterio.Affine(1,0,0,0,1,0) #Transform is just an identity matrix for a JPEG image
            })
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(window_data)
            print(f"Saved window to {output_path}")
                
windows = chop2nWindows(img_path, n)
print(f"Generated {len(windows)} windows.")
for i, window in enumerate(windows):
    print(window)
    output_path = out_folder_path + f"window_{i}.jpg"
    saveWindow(img_path, window, output_path)
