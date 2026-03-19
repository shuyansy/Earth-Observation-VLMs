import os
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from tqdm import tqdm

# 输入根目录和输出根目录
input_root = '/data/BigEarthNet-S2'
output_root = '/data/bigearthnet_data'
os.makedirs(output_root, exist_ok=True)

# Sentinel-2 波段顺序
sentinel_band_order = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                       'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

# 提取波段名（B01、B8A 等）
def extract_band_name(filename):
    match = re.search(r'_B(\d{1,2}A?)\.tif$', filename)
    return 'B' + match.group(1) if match else None

# 遍历底层的包含 .tif 文件的文件夹
for root, dirs, files in os.walk(input_root):
    tif_files = [f for f in files if f.endswith('.tif')]
    if len(tif_files) < 12:
        continue  # 跳过不完整样本
    
    # 提取当前子目录名作为输出目录名
    rel_path = os.path.relpath(root, input_root)
    out_dir = os.path.join(output_root, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    # 排序波段文件
    tif_files = sorted(tif_files, key=lambda x: sentinel_band_order.index(extract_band_name(x)))

    # 使用 B04 作为参考分辨率
    ref_file = [f for f in tif_files if extract_band_name(f) == 'B04'][0]
    ref_path = os.path.join(root, ref_file)
    with rasterio.open(ref_path) as ref_src:
        target_shape = (ref_src.height, ref_src.width)

    # 每3个波段合成为一张 RGB
    for i in range(0, len(tif_files), 3):
        group = tif_files[i:i+3]
        if len(group) < 3:
            break  # 最后一组不足3个波段不处理

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
