import numpy as np
from s2cloudless import S2PixelCloudDetector
from sklearn.metrics import mean_squared_error

from .process import get_detect_cloud_bands


def get_mask_crop_correlation(cloudy_bands, cloudy_mask):
    bands = {}
    for band in range(cloudy_bands.shape[-1]):
        cloudy_crop = cloudy_bands[:, :, band][cloudy_mask > 0.5]
        mask_crop = cloudy_mask[cloudy_mask > 0.5]
        if mask_crop.shape == (0,):
            bands[f"b{band}_cloudy_crop_correlation_mse"] = -1
        else:
            bands[f"b{band}_cloudy_crop_correlation_mse"] = mean_squared_error(cloudy_crop, mask_crop)
        bands[f"b{band}_cloudy_band_correlation_mse"] = mean_squared_error(cloudy_bands[:, :, band].reshape(-1),
                                                                           cloudy_mask.reshape(-1))
    return bands

def get_correlation_cloud_bands(cloud_detector, cloudly_img):
    cloudy_bands = get_detect_cloud_bands(cloudly_img)
    cloudly_mask = cloud_detector.compute_mask_cloud(cloudy_bands)
    return get_mask_crop_correlation(cloudy_bands, cloudly_mask)

class CloudDetector:

    def __init__(self):
        self.model = S2PixelCloudDetector(threshold=0.6, average_over=4, dilation_size=2, all_bands=True)

    def compute_percentage_cloud(self, img):
        cloudy_bands = get_detect_cloud_bands(img)
        cloudy_mask = self.model.get_cloud_probability_maps(cloudy_bands[np.newaxis, ...])
        return cloudy_mask.mean()

    def compute_mask_cloud(self, cloudy_bands):
        return self.model.get_cloud_masks(cloudy_bands[np.newaxis, ...])[0]


