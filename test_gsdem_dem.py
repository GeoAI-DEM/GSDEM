import os
import torch
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from sklearn import metrics
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from osgeo import gdal
import cv2

from models.network_gsdem import GSDEM


# Normalization parameters
NORM_OFFSET = 2149.0
NORM_SCALE = 19711.0

def normalize_dem(data):
    """Normalize DEM data to [-1, 1] range"""
    return (2.0 * data + NORM_OFFSET) / NORM_SCALE

def denormalize_dem(data):
    """Denormalize from [-1, 1] range back to original scale"""
    return (data * NORM_SCALE - NORM_OFFSET) / 2.0


def compare(pred, gt):
    """Calculate metrics: RMSE, MAE, R2, ME, percent1, percent2"""
    dstflat = gt.flatten()
    imgconpareflat = pred.flatten()

    mse = metrics.mean_squared_error(dstflat, imgconpareflat)
    rmse = mse ** 0.5
    mae = metrics.mean_absolute_error(dstflat, imgconpareflat)
    r2 = metrics.r2_score(dstflat, imgconpareflat)

    error = dstflat - imgconpareflat
    error_abs = np.abs(error)
    error_sum = np.sum(error)
    me = error_sum / len(error)

    # Percentage of errors < 20
    num = np.sum(error_abs < 20)
    percent1 = num / len(error)

    # Percentage of errors < 10
    num2 = np.sum(error_abs < 10)
    percent2 = num2 / len(error)

    return mse, rmse, mae, r2, me, percent1, percent2


def AddRound(npgrid):
    """Add padding border to image"""
    nx, ny = npgrid.shape[0], npgrid.shape[1]
    zbc = np.zeros((nx + 20, ny + 20))

    # Fill center
    zbc[10:-10, 10:-10] = npgrid

    # Fill edges
    zbc[0:10, 10:-10] = npgrid[0:10, :]
    zbc[-10:, 10:-10] = npgrid[-10:, :]
    zbc[10:-10, 0:10] = npgrid[:, 0:10]
    zbc[10:-10, -10:] = npgrid[:, -10:]

    # Fill corners
    zbc[0:10, 0:10] = npgrid[0:10, 0:10]
    zbc[0:10, -10:] = npgrid[0:10, -10:]
    zbc[-10:, 0:10] = npgrid[-10:, 0:10]
    zbc[-10:, -10:] = npgrid[-10:, -10:]

    return zbc


def read_tiff(inpath):
    """Read TIFF file with geotransform and projection"""
    dataset = gdal.Open(inpath)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_proj, im_geotrans, im_data


def caculatemodel(path500m, path125m, pathadd125m, model, sizeinput, log_file=None):
    """Process image with block-wise inference"""
    # Read LR image
    ds = gdal.Open(path500m, gdal.GA_ReadOnly)
    img500 = np.array(ds.GetRasterBand(1).ReadAsArray())
    im_input = normalize_dem(img500)

    colum = im_input.shape[1]
    row = im_input.shape[0]
    size = sizeinput
    colum_size = colum // size
    row_size = row // size

    # Output array for SR
    dataout = np.zeros(colum * row * 225, dtype=np.float32).reshape(row * 15, colum * 15)
    add_im_input = AddRound(im_input)

    model = model.cuda()
    device = torch.device('cuda')

    k = 0
    for i in range(row_size):
        inx = i * size
        for j in range(colum_size):
            iny = j * size
            imputmodel = add_im_input[inx:inx + size + 20, iny:iny + size + 20]

            # Convert to tensor (1 channel for GSDEM)
            im_inputvar = torch.from_numpy(imputmodel).float().unsqueeze(0).unsqueeze(0)

            # Inference
            with torch.no_grad():
                HR = model(im_inputvar.to(device))

            HR = HR.cpu().numpy().astype(np.float32)

            # Extract center region (remove padding of 150 pixels on each side, which is 150*15=2250 at 15x scale)
            # But since the input patch is (size+20) normalized, and we're doing 15x upsampling
            # The output is (size+20)*15, and we want to remove 150 from each side
            sr_output = HR[0, 0, 150:-150, 150:-150]

            dataout[inx * 15:inx * 15 + size * 15, iny * 15:iny * 15 + size * 15] = sr_output
            k = k + 1

    # Denormalize
    dataout = denormalize_dem(dataout)

    # Read HR ground truth
    ds = gdal.Open(path125m, gdal.GA_ReadOnly)
    img125 = np.array(ds.GetRasterBand(1).ReadAsArray())

    # Bicubic upsampling baseline
    fx = 15
    fy = 15
    img = np.dstack([img500] * 3)
    enlarge_CUBIC = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    enlarge_CUBIC = enlarge_CUBIC[:, :, 1]

    # Calculate metrics for SR
    MSE, RMSE, MAE, R2, ME, percent1, percent2 = compare(img125, dataout)
    sr_output = pathadd125m[:-9] + "_SR: " + f"RMSE={RMSE:.6f} MAE={MAE:.6f} R2={R2:.6f} ME={ME:.6f} percent1={percent1:.6f} percent2={percent2:.6f}"
    print(sr_output)
    if log_file:
        log_file.write(sr_output + '\n')

    # Calculate metrics for bicubic
    MSE, RMSE, MAE, R2, ME, percent1, percent2 = compare(img125, enlarge_CUBIC)
    bicubic_output = pathadd125m[:-9] + "_bicubic: " + f"RMSE={RMSE:.6f} MAE={MAE:.6f} R2={R2:.6f} ME={ME:.6f} percent1={percent1:.6f} percent2={percent2:.6f}"
    print(bicubic_output)
    if log_file:
        log_file.write(bicubic_output + '\n')

    # Calculate PSNR and SSIM on normalized data
    img125nor = normalize_dem(img125)
    dataoutnor = normalize_dem(dataout)
    enlarge_CUBICnor = normalize_dem(enlarge_CUBIC)

    psnr_sr = peak_signal_noise_ratio(img125nor, dataoutnor, data_range=1.0)
    ssim_sr = structural_similarity(img125nor, dataoutnor, data_range=1.0)

    psnr_bicubic = peak_signal_noise_ratio(img125nor, enlarge_CUBICnor, data_range=1.0)
    ssim_bicubic = structural_similarity(img125nor, enlarge_CUBICnor, data_range=1.0)

    psnr_output = pathadd125m[:-9] + '_psnr_SR_ssim_SR: ' + f"PSNR={psnr_sr:.6f} SSIM={ssim_sr:.6f}"
    print(psnr_output)
    if log_file:
        log_file.write(psnr_output + '\n')

    bicubic_psnr_output = pathadd125m[:-9] + '_psnr_bicubic_ssim_bicubic: ' + f"PSNR={psnr_bicubic:.6f} SSIM={ssim_bicubic:.6f}"
    print(bicubic_psnr_output)
    if log_file:
        log_file.write(bicubic_psnr_output + '\n')
        log_file.flush()


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Initialize model
model = GSDEM(
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
).to(device)

model = nn.DataParallel(model)

# Load checkpoint
checkpoint_path = "your path"
print(f"Loading model from {checkpoint_path}")
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("✓ Model loaded successfully")
else:
    print(f"✗ Checkpoint not found at {checkpoint_path}")
    print("Please train the model first")
    exit(1)

# Test data path
pathbase = "your path"
csv_path = "your path"

# Find all LR files
lr_files = sorted([f for f in os.listdir(pathbase) if f.endswith('_500m.tif')])
print(f"Found {len(lr_files)} LR files\n")

# Open CSV file
csv_file = open(csv_path, 'w', encoding='utf-8')

# Test each image pair
for idx, lr_file in enumerate(lr_files):
    # Get base name and HR file name
    base_name = lr_file.replace('_500m.tif', '')
    hr_file = f"{base_name}_30m.tif"

    path500m = os.path.join(pathbase, lr_file)
    path125m = os.path.join(pathbase, hr_file)

    # Check if HR file exists
    if not os.path.exists(path125m):
        warn_msg = f"Warning: Cannot find HR file {hr_file}"
        print(warn_msg)
        csv_file.write(warn_msg + '\n')
        csv_file.flush()
        continue

    # Get image size and calculate block size
    ds = gdal.Open(path500m, gdal.GA_ReadOnly)
    if ds is None:
        error_msg = f"Error: Cannot open {lr_file}"
        print(error_msg)
        csv_file.write(error_msg + '\n')
        csv_file.flush()
        continue

    width = ds.RasterXSize
    height = ds.RasterYSize
    del ds

    sizeinput = max(width, height) // 6

    progress_msg = f"\n[{idx+1}/{len(lr_files)}] Processing: {base_name}"
    print(progress_msg)
    csv_file.write(progress_msg + '\n')
    csv_file.flush()

    try:
        caculatemodel(path500m, path125m, hr_file, model, sizeinput, csv_file)
    except Exception as e:
        error_msg = f"Error processing: {e}"
        print(error_msg)
        csv_file.write(error_msg + '\n')
        csv_file.flush()

csv_file.close()
finish_msg = f"\nDone! Results saved to: {csv_path}"
print(finish_msg)


