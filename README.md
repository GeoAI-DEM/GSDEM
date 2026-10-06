# GSDEM

GSDEM is a super-resolution project for Digital Elevation Models (DEM). It reconstructs a
high-resolution single-channel DEM from a low-resolution input. The current model performs
15× upscaling using a 3× then 5× staged upsampling pipeline.


## GLOD-1s Dataset

GLOD-1s is the global 1 arc-second land–ocean DEM generated using the GSDEM framework. The complete dataset is approximately 2.4 TB.

Due to its large data volume, GLOD-1s is not hosted directly in this GitHub repository. The complete dataset will be distributed through Google Drive. Public download links and access instructions will be added to this repository upon acceptance of the associated manuscript.

**Current status:** Preparing for public release.

## Repository Layout

- `models/network_gsdem.py` - Model definition and optional domain classifier
- `dataprocess.py` - Example data preprocessing and HDF5 dataset builder
- `infer_gebco_global.py` - Global inference script for GEBCO 2025 (15× super-resolution)
- `train_gsdem_dem.py` - Training (pretrain/finetune)
- `test_gsdem_dem.py` - GeoTIFF testing with GDAL and block-wise inference
- `utils/util_calculate_psnr_ssim.py` - Metrics (PSNR/SSIM/PSNR-B)
- `stop_utils.py` - EarlyStopping utility

## Environment

```
pip install torch torchvision timm numpy opencv-python scikit-learn h5py scikit-image
```

For GeoTIFF testing:

```
pip install GDAL
```

## Data Format

Training uses HDF5 files with the following datasets:

- `32_32`: low-resolution DEM patches (N, 1, 32, 32)
- `480_480`: high-resolution DEM patches (N, 1, 480, 480)

## Training

1) Configure paths and options at the top of `train_gsdem_dem.py`
2) Run:

```
python train_gsdem_dem.py
```

## Testing

1) Configure paths in `test_gsdem_dem.py`
2) Run:

```
python test_gsdem_dem.py
```

Outputs RMSE, MAE, R2, ME, PSNR, and SSIM, and writes a CSV log.

## Data Preprocessing

`dataprocess.py` is an example script for cropping raster data and building HDF5
training sets. Adjust paths, thresholds, and region filters to your dataset.

## Global Inference

`infer_gebco_global.py` performs 15× super-resolution on GEBCO 2025 and writes
NASADEM-style tiles. Configure GEBCO path, checkpoint path, and output directory
in the script header before running.

## Notes

- Update the hard-coded example paths in training/testing/inference scripts to match your environment.
- The global inference script assumes a 1°×1° tiling scheme and GEBCO 2025 grid specs.

## Citation

If you use this repository in academic work, please cite your paper or project
containing GSDEM details.
