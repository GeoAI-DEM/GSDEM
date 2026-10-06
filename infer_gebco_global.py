"""
Global GEBCO2025 Super-resolution Inference
15x super-resolution: 240×240 (15 arcsec) → 3600×3600 (1 arcsec)
Saves output in NASADEM format (nXXeYYY.tif)
"""


import os
import sys
import torch
import torch.nn as nn
import numpy as np
from osgeo import gdal, osr
import warnings
warnings.filterwarnings('ignore')

# Add model path
sys.path.insert(0, 'your path')
from models.network_swinir import SwinIR

# ============================================================
# Configuration (User should modify paths here)
# ============================================================
GEBCO_PATH = "your path"                              # Set by user: path to GEBCO_2025.tif
CHECKPOINT_PATH = "your path"                        # Set by user: path to trained model checkpoint
OUTPUT_DIR = "your path"                              # Set by user: output directory for results
NASADEM_TRAIN_DIR = "your path"          # NASADEM training data folder
NASADEM_VAL_DIR = "your path"              # NASADEM validation data folder
DEVICE_IDS = [0, 1]                            # GPU IDs for DataParallel
LOG_FILE = "your path"                                 # Set by user: path to log file (optional)

# Normalization parameters (from test_swinir_dem.py)
NORM_OFFSET = 2149.0
NORM_SCALE = 19711.0

# GEBCO2025 parameters
GEBCO_PIXEL_SIZE = 0.004166666666667          # 15 arcsec in degrees
GEBCO_ORIGIN_LON = -180.0
GEBCO_ORIGIN_LAT = 90.0
GEBCO_WIDTH = 86400
GEBCO_HEIGHT = 43200

# Output parameters (NASADEM format)
OUTPUT_PIXEL_SIZE = 0.000277777777778         # 1 arcsec in degrees
OUTPUT_SIZE = 3600                             # 3600×3600 pixels per 1°×1° block

# ============================================================
# Validation
# ============================================================
def validate_config():
    """Validate user configuration"""
    if GEBCO_PATH is None:
        raise ValueError("GEBCO_PATH not set! Please modify the script.")
    if CHECKPOINT_PATH is None:
        raise ValueError("CHECKPOINT_PATH not set! Please modify the script.")
    if OUTPUT_DIR is None:
        raise ValueError("OUTPUT_DIR not set! Please modify the script.")

    if not os.path.exists(GEBCO_PATH):
        raise FileNotFoundError(f"GEBCO file not found: {GEBCO_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Configuration validated")
    print(f"  GEBCO: {GEBCO_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    if LOG_FILE:
        print(f"  Log: {LOG_FILE}")


# ============================================================
# Logging utility
# ============================================================
def log_and_print(message, log_file=None):
    """Print to console and optionally write to log file"""
    print(message)
    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            print(f"⚠ Error writing to log file: {e}")


# ============================================================
# Data normalization
# ============================================================
def normalize_dem(data):
    """Normalize DEM data to [-1, 1] range (same as training)"""
    return (2.0 * data + NORM_OFFSET) / NORM_SCALE


def denormalize_dem(data):
    """Denormalize from [-1, 1] range back to original scale"""
    return (data * NORM_SCALE - NORM_OFFSET) / 2.0


# ============================================================
# Coordinate conversion
# ============================================================
def latlon_to_gebco_pixel(lat, lon):
    """Convert geographic coordinates (lat, lon) to GEBCO2025 pixel indices (精确对齐)"""
    # 不处理国际日期变更线，直接计算
    # （日期变更线 180/-180 在这里是边界情况，正常计算即可）

    # 用round()处理浮点精度，确保相邻块对齐
    col = int(round((lon - GEBCO_ORIGIN_LON) / GEBCO_PIXEL_SIZE))
    row = int(round((GEBCO_ORIGIN_LAT - lat) / GEBCO_PIXEL_SIZE))
    return row, col


def extract_gebco_block(gebco_data, lat, lon):
    """
    Extract 240×240 block from GEBCO2025 for 1°×1° tile at (lat, lon)

    Args:
        gebco_data: Full GEBCO2025 array
        lat: Starting latitude (south edge of tile)
        lon: Starting longitude (west edge of tile)

    Returns:
        block: 240×240 array (or smaller if at boundary)
    """
    row_min, col_min = latlon_to_gebco_pixel(lat + 1.0, lon)  # North edge
    row_max, col_max = latlon_to_gebco_pixel(lat, lon + 1.0)  # South edge

    # Clamp to valid GEBCO range
    row_min = max(0, min(row_min, GEBCO_HEIGHT))
    row_max = max(0, min(row_max, GEBCO_HEIGHT))
    col_min = max(0, min(col_min, GEBCO_WIDTH))
    col_max = max(0, min(col_max, GEBCO_WIDTH))

    block = gebco_data[row_min:row_max, col_min:col_max]
    return block


def format_nasadem_name(lat, lon):
    """
    Format NASADEM filename from coordinates
    Examples: n00e009.tif, s30w060.tif
    """
    # Latitude part
    lat_int = int(lat)
    if lat >= 0:
        lat_str = f"n{abs(lat_int):02d}"
    else:
        lat_str = f"s{abs(lat_int):02d}"

    # Longitude part
    lon_int = int(lon)
    if lon >= 0:
        lon_str = f"e{abs(lon_int):03d}"
    else:
        lon_str = f"w{abs(lon_int):03d}"

    return f"{lat_str}{lon_str}.tif"


# ============================================================
# Model inference
# ============================================================
def load_model(checkpoint_path, device_ids):
    """Load trained SwinIR model"""
    model = SwinIR(
        upscale=15,
        in_chans=1,
        img_size=32,
        window_size=8,
        img_range=1.,
        depths=[12, 12, 12, 12],
        embed_dim=180,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        use_gated_mlp_in_last_layer=True
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle DataParallel checkpoint (remove 'module.' prefix if present)
    if 'module.' in next(iter(checkpoint.keys())):
        checkpoint = {k.replace('module.', '', 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    print(f"✓ Model loaded from {checkpoint_path}")

    # Move to GPU and DataParallel
    device = torch.device(f'cuda:{device_ids[0]}')
    model = model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    print(f"✓ Model on GPU(s): {device_ids}")
    return model, device


def superresolve_batch(blocks, model, device):
    """
    Super-resolve batch of 240×240 blocks to 3600×3600
    处理NaN（NoData标准）

    Args:
        blocks: List of 240×240 arrays (may contain NaN)
        model: SwinIR model
        device: torch device

    Returns:
        results: List of 3600×3600 super-resolved arrays
    """
    batch_size = len(blocks)
    results = []

    for block in blocks:
        # 记录NaN位置
        nodata_mask = np.isnan(block)

        # 将NaN临时替换为0用于归一化（NaN会导致计算问题）
        block_clean = np.nan_to_num(block, nan=0.0)

        # Normalize
        block_norm = normalize_dem(block_clean.astype(np.float32))

        # Convert to tensor: (1, 1, H, W)
        block_tensor = torch.from_numpy(block_norm[np.newaxis, np.newaxis, :, :]).to(device)

        # Inference
        with torch.no_grad():
            output_tensor = model(block_tensor)

        # Convert back to numpy
        output = output_tensor.squeeze().cpu().numpy()

        # Denormalize
        output_dem = denormalize_dem(output)

        # 处理NaN区域：超分后的对应位置也设为NaN
        if nodata_mask.any():
            # NaN 15倍扩展（240→3600）
            nodata_mask_expanded = np.repeat(np.repeat(nodata_mask, 15, axis=0), 15, axis=1)
            # Clip to exact output size
            nodata_mask_expanded = nodata_mask_expanded[:output_dem.shape[0], :output_dem.shape[1]]
            # Set NoData regions to NaN
            output_dem[nodata_mask_expanded] = np.nan

        results.append(output_dem.astype(np.float32))

    return results


# ============================================================
# Merge with NASADEM
# ============================================================
def merge_with_nasadem(sr_data, lat, lon, train_dir, val_dir, target_size=3600):
    """
    Merge strategy (3600×3600版本，无resize):
    - sr_data: 3600×3600超分结果
    - NASADEM有有效数据（!=−32768）→ 陆地，用NASADEM的3600×3600中心部分
    - NASADEM无数据或不存在 → 海洋，用超分值
    - NoData统一为NaN

    Args:
        sr_data: Super-resolved array (3600×3600)
        lat, lon: Tile coordinates
        train_dir: NASADEM train folder path
        val_dir: NASADEM val folder path
        target_size: Output size (default 3600, no resize)

    Returns:
        merged_data: Merged array (3600×3600)
        has_nasadem: Boolean, whether NASADEM was found and merged
        replacement_ratio: Float, percentage of pixels replaced from NASADEM (0-1)
    """
    # 1. 构建NASADEM文件名
    nasadem_filename = format_nasadem_name(lat, lon)

    # 2. 尝试在train和val文件夹中找NASADEM文件
    nasadem_path = None
    if os.path.exists(os.path.join(train_dir, nasadem_filename)):
        nasadem_path = os.path.join(train_dir, nasadem_filename)
    elif os.path.exists(os.path.join(val_dir, nasadem_filename)):
        nasadem_path = os.path.join(val_dir, nasadem_filename)

    # 3. 如果找不到NASADEM，直接返回超分结果
    if nasadem_path is None:
        # 海洋区域，无NASADEM，返回纯超分数据
        return sr_data, False, 0.0

    # 4. 读取NASADEM（3601×3601）
    try:
        nasadem_ds = gdal.Open(nasadem_path)
        nasadem_height = nasadem_ds.RasterYSize
        nasadem_width = nasadem_ds.RasterXSize
        nasadem_band = nasadem_ds.GetRasterBand(1)
        nasadem_data = nasadem_band.ReadAsArray().astype(np.float32)
        nasadem_ds = None  # Close dataset
    except Exception as e:
        print(f"⚠ Error reading NASADEM {nasadem_filename}: {str(e)}")
        # 降级处理：返回超分数据
        return sr_data, False, 0.0

    # 5. 从NASADEM（3601×3601）提取中心3600×3600
    # 方法：去掉最后一行和最后一列
    nasadem_crop = nasadem_data[:3600, :3600]

    # 6. 关键合并逻辑：只替换超分结果>0且NASADEM有数据的像素
    # NASADEM的NoData值是-32768
    valid_nasadem_mask = (sr_data > 0) & (nasadem_crop != -32768)
    sr_data[valid_nasadem_mask] = nasadem_crop[valid_nasadem_mask]

    # 计算替换比例
    replacement_count = np.sum(valid_nasadem_mask)
    total_pixels = sr_data.size
    replacement_ratio = replacement_count / total_pixels

    # 7. 清理NoData：把NASADEM的-32768改成NaN（统一NoData标准）
    sr_data[nasadem_crop == -32768] = np.nan

    return sr_data, True, replacement_ratio


# ============================================================
# File I/O
# ============================================================
def save_nasadem_tif(data, output_path, lat, lon):
    """
    Save block with NaN as NoData and precise geotransform for seamless tiling

    Args:
        data: 3600×3600 array
        output_path: Output file path
        lat: Starting latitude
        lon: Starting longitude
    """
    height, width = data.shape

    # Create driver with LZW compression options
    driver = gdal.GetDriverByName('GTiff')
    options = ['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Float32, options=options)

    # 关键：精确地理变换
    # 块的左上角坐标 (west, north)
    # 确保相邻块的边界完全重合
    west = float(lon)
    north = float(lat + 1.0)

    geotransform = (
        west,                    # west edge
        OUTPUT_PIXEL_SIZE,       # pixel width
        0.0,
        north,                   # north edge
        0.0,
        -OUTPUT_PIXEL_SIZE       # pixel height
    )
    dataset.SetGeoTransform(geotransform)

    # Set projection (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    dataset.SetProjection(srs.ExportToWkt())

    # Write data
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    # NaN自动被GDAL识别为NoData
    band.SetNoDataValue(np.nan)
    band.FlushCache()

    # Close dataset
    dataset = None


# ============================================================
# Main processing
# ============================================================
def main():
    """Process all 1°×1° tiles"""
    validate_config()

    # Initialize log file if specified
    if LOG_FILE:
        try:
            with open(LOG_FILE, 'w') as f:
                f.write(f"GEBCO2025 Super-resolution Inference Log\n")
                f.write(f"{'='*60}\n")
        except Exception as e:
            print(f"⚠ Error initializing log file: {e}")

    # Load GEBCO data
    print("\nLoading GEBCO2025 data...")
    gebco = gdal.Open(GEBCO_PATH)
    gebco_band = gebco.GetRasterBand(1)
    gebco_data = gebco_band.ReadAsArray().astype(np.float32)
    print(f"✓ GEBCO loaded: {gebco_data.shape}")

    # Load model
    print("\nLoading model...")
    model, device = load_model(CHECKPOINT_PATH, DEVICE_IDS)

    # Process tiles with batch accumulation
    print("\nProcessing tiles (batch mode with 2 GPUs)...")
    total_tiles = 180 * 360  # All possible 1°×1° tiles
    processed = 0
    failed = 0

    batch_size = 8  # Accumulate 8 blocks per batch for 2 GPUs
    batch_blocks = []
    batch_info = []  # Store (lat, lon, filename) for each block in batch

    for lat in range(-90, 90):
        for lon in range(-180, 180):
            try:
                # Extract block
                block = extract_gebco_block(gebco_data, lat, lon)

                # Skip empty blocks
                if block.size == 0:
                    continue

                # Pad if needed (should be 240×240)
                if block.shape != (240, 240):
                    # Pad with NaN
                    padded_block = np.full((240, 240), np.nan, dtype=np.float32)
                    h, w = block.shape
                    padded_block[:h, :w] = block
                    block = padded_block

                # Accumulate in batch
                batch_blocks.append(block)
                filename = format_nasadem_name(lat, lon)
                batch_info.append((lat, lon, filename))

                # Process batch when full or at end of loop
                if len(batch_blocks) >= batch_size or (lat == 89 and lon == 179):
                    # Super-resolve batch
                    results = superresolve_batch(batch_blocks, model, device)

                    # Save all results in batch
                    for (batch_lat, batch_lon, batch_filename), result in zip(batch_info, results):
                        # 与NASADEM合并（3600×3600，无resize）
                        result, has_nasadem, replacement_ratio = merge_with_nasadem(
                            result, batch_lat, batch_lon,
                            NASADEM_TRAIN_DIR, NASADEM_VAL_DIR,
                            target_size=3600
                        )

                        output_path = os.path.join(OUTPUT_DIR, batch_filename)
                        save_nasadem_tif(result, output_path, batch_lat, batch_lon)

                        # 记录日志
                        if has_nasadem:
                            msg = f"✓ Saved: {batch_filename} (NASADEM replacement: {replacement_ratio*100:.1f}%)"
                        else:
                            msg = f"✓ Saved: {batch_filename} (SR-only, no NASADEM)"
                        log_and_print(msg, LOG_FILE)
                        processed += 1

                    if processed % 100 == 0:
                        print(f"  Processed {processed}/{total_tiles} tiles")

                    # Clear batch
                    batch_blocks = []
                    batch_info = []

            except Exception as e:
                failed += 1
                print(f"✗ Error at ({lat}, {lon}): {str(e)}")

    print(f"\n{'='*60}")
    print(f"✓ Global GEBCO2025 Super-resolution Inference Complete!")
    print(f"{'='*60}")
    print(f"Total tiles processed: {processed}")
    print(f"Failed tiles: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
