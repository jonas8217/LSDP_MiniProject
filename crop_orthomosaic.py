import rasterio
from rasterio.windows import Window
from rasterio.env import Env

in_folder_path = "./orthomosaic/Gyldensteensvej-9-19-2017-orthophoto.jpg"
out_folder_path = "./orthomosaic/"
mosaic_name = "pumpkin_patch_cut_better.jpg"
img_path = out_folder_path + mosaic_name


with Env(GDAL_PAM_ENABLED='NO'):    #Disable the creation of an .aux.xml file for each .jpg file
    with rasterio.open(in_folder_path) as src:
        col_off = src.width / 14
        row_off = src.height / 3.5
        width = src.width * 0.8
        height = src.height * 0.3
        window = Window(col_off, row_off, width, height)

        window_data = src.read(window=window)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "JPEG",
            "height": window.height,
            "width": window.width
        })
        with rasterio.open(img_path, "w", **out_meta) as dest:
            dest.write(window_data)
        print(f"Saved window to {img_path}")
