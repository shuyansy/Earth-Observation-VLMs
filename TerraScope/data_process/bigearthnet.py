import os
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from tqdm import tqdm

input_root = 'data/bigearthnet/BigEarthNet-S2'
output_root = 'data/bigearthnet/output'
os.makedirs(output_root, exist_ok=True)

sentinel_band_order = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                       'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

def extract_band_name(filename):
    match = re.search(r'_B(\d{1,2}A?)\.tif$', filename)
    return 'B' + match.group(1) if match else None

for root, dirs, files in os.walk(input_root):
    tif_files = [f for f in files if f.endswith('.tif')]
    if len(tif_files) < 12:
        continue
    
    rel_path = os.path.relpath(root, input_root)
    out_dir = os.path.join(output_root, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(tif_files, key=lambda x: sentinel_band_order.index(extract_band_name(x)))

    ref_file = [f for f in tif_files if extract_band_name(f) == 'B04'][0]
    ref_path = os.path.join(root, ref_file)
    with rasterio.open(ref_path) as ref_src:
        target_shape = (ref_src.height, ref_src.width)

    for i in range(0, len(tif_files), 3):
        group = tif_files[i:i+3]
        if len(group) < 3:
            break

        channels = []
        for fname in group:
            with rasterio.open(os.path.join(root, fname)) as src:
                data = src.read(
                    1,
                    out_shape=target_shape,
                    resampling=Resampling.bilinear
                )
                channels.append(data)

        rgb = np.stack(channels, axis=-1)  # (H, W, 3)
        rgb = np.clip(rgb, 0, 3000) / 3000.0
        rgb = (rgb * 255).astype(np.uint8)

        img = Image.fromarray(rgb)
        out_name = f"rgb_{i//3 + 1}.png"
        img.save(os.path.join(out_dir, out_name))
    
    print(f"✅ Processed: {rel_path}")
