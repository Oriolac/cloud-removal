import cv2
import numpy as np
from scipy.stats import gaussian_kde
from skimage.metrics import mean_squared_error

from .process import process_bands


def compute_contrast_score(band):
    # Compute the standard deviation of each band
    stds = np.std(band, axis=(0, 1))
    mean_std = np.mean(stds)

    # Compute the contrast score
    contrast = mean_std / np.mean(band)
    # Return the contrast score
    return contrast


def compute_michelson_contrast(band):
    min_val, max_val, _, _ = cv2.minMaxLoc(band)
    michelson_contrast = (max_val - min_val) / (max_val + min_val)
    return michelson_contrast


def compute_rms_contrast(band):
    mean_val = np.mean(band)
    rms_contrast = np.sqrt(np.mean((band - mean_val) ** 2))
    return rms_contrast


def compute_laplacian_blur(band):
    laplacian = cv2.Laplacian(band, cv2.CV_64F)
    variance = laplacian.var()
    return variance



def compute_band_properties(img):
    img = process_bands(img)
    res = {}
    for b in range(img.shape[-1]):
        res[f"{b}_traditional_contrast"] = compute_contrast_score(img[:,:,b])
        res[f"{b}_michelson_contrast"] = compute_michelson_contrast(img[:,:,b])
        res[f"{b}_rms_contrast"] = compute_rms_contrast(img[:,:,b])
        res[f"{b}_laplacian_blur"] = compute_laplacian_blur(img[:, :, b])
        res[f'{b}_mean'] = img[:, :, b].mean()
        res[f'{b}_median'] = np.percentile(img[:, :, b], 50)
        res[f'{b}_p25'] = np.percentile(img[:, :, b], 25)
        res[f'{b}_p75'] = np.percentile(img[:, :, b], 75)
        res[f'{b}_std'] = img[:, :, b].std()
    return res


def compute_paired_mse_bands(cloudless_img, cloudly_img):
    cloudless = process_bands(cloudless_img)
    cloudy = process_bands(cloudly_img)
    res = {}
    for b in range(cloudless.shape[-1]):
        res[f"b{b}_mse_paired"] = mean_squared_error(cloudless[:, :, b], cloudy[:, :, b])
    return res


def compute_kde_band(band):
    data = band.ravel()
    kde = gaussian_kde(data)
    x_vals = np.linspace(0, 255, 80)
    y_vals = kde(x_vals)
    return y_vals


def compute_kde_patch(bands):
    img = process_bands(bands)
    res = {}
    for band in range(13):
        res[f"b{band} kde"] = np.array(compute_kde_band(img[:, :, band]))
    return res