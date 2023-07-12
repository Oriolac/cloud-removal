import cv2
import numpy as np
from skimage.filters import threshold_otsu

from .process import get_tci, tci_names


def mse_component(otsu_hold):
    mean_cols = otsu_hold.sum(0).mean()
    median_cols = np.percentile(otsu_hold.sum(0), 50)
    mean_rows = otsu_hold.sum(1).mean()
    median_rows = np.percentile(otsu_hold.sum(1), 50)
    return {
        "hidden_cols_mean": ((mean_cols - otsu_hold.sum(0)) ** 2).mean(),
        "hidden_cols_median": ((median_cols - otsu_hold.sum(0)) ** 2).mean(),
        "hidden_rows_mean": ((mean_rows - otsu_hold.sum(1)) ** 2).mean(),
        "hidden_rows_median": ((median_rows - otsu_hold.sum(1)) ** 2).mean(),
    }


def get_umbralization(img):
    mses = {}
    otsus = []
    for b, bname in enumerate(tci_names()):
        otsu = threshold_otsu(img[:, :, b])
        binarized = img[:, :, b] > otsu
        for mse_key, mse_value in mse_component(binarized).items():
            mses[f"{bname}_{mse_key}"] = mse_value
        otsus.append(binarized)
    return mses


def compute_saturation(image):
    # Convert the image from RGB to HSV color space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract the saturation channel
    sat = img_hsv[:, :, 1]

    return sat.mean()


def compute_color_temperature(image):
    # Convert the image from RGB to CIE 1931 XYZ color space
    img_xyz = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)

    # Calculate the chromaticity coordinates
    sum_xyz = np.sum(img_xyz, axis=(0, 1))
    x = sum_xyz[0] / np.sum(sum_xyz)
    y = sum_xyz[1] / np.sum(sum_xyz)

    # Calculate the correlated color temperature (CCT)
    n = (x - 0.332) / (0.1858 - y)
    cct = 449 * (n ** 3) + 3525 * (n ** 2) + 6823.3 * n + 5520.33

    return cct


def get_mse_hidden_object(tci_img):
    return get_umbralization(tci_img)


def compute_unique_properties(img):
    tci_img = get_tci(img)
    res = get_mse_hidden_object(tci_img)
    res['saturation'] = compute_saturation(tci_img)
    res['temperature'] = compute_color_temperature(tci_img)
    return res
