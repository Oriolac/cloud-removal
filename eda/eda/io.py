import dataclasses
import os
import pathlib

import rasterio as rio

from .bands import compute_band_properties, compute_paired_mse_bands, compute_kde_patch
from .cloud import CloudDetector, get_mask_crop_correlation
from .process import get_detect_cloud_bands
from .unique import compute_unique_properties


@dataclasses.dataclass
class Patch:
    roi: str
    season: str
    scenario: int
    patch: int
    type: str
    path: pathlib.Path


def add_patch_data(data, patch: Patch):
    data['roi'] = patch.roi
    data['season'] = patch.season
    data['patch'] = patch.patch
    data['scene'] = patch.scenario


def map_func(cloudless: Patch, cloudy: Patch):
    assert cloudless.season == cloudy.season, f"{cloudless} {cloudy}"
    assert cloudless.roi == cloudy.roi, f"{cloudless} {cloudy}"
    assert cloudless.patch == cloudy.patch, f"{cloudless} {cloudy}"

    cloud_detector = CloudDetector()
    cloudless_img = rio.open(cloudless.path)
    cloudy_img = rio.open(cloudy.path)
    data_cloudless = computation_solo_properties(cloudless_img, cloud_detector)
    data_cloudy = computation_solo_properties(cloudy_img, cloud_detector)
    cloudy_bands = get_detect_cloud_bands(cloudy_img)
    cloudy_mask = cloud_detector.compute_mask_cloud(cloudy_bands)
    data_cloudy = dict(**data_cloudy, **get_mask_crop_correlation(cloudy_bands, cloudy_mask))

    data_paired = compute_paired_mse_bands(cloudless_img, cloudy_img)
    cloudless_kde = compute_kde_patch(cloudless_img)
    cloudy_kde = compute_kde_patch(cloudy_img)

    add_patch_data(data_cloudless, cloudless)
    add_patch_data(data_cloudy, cloudy)
    add_patch_data(data_paired, cloudless)

    return data_cloudless, data_cloudy, data_paired, cloudless_kde, cloudy_kde, cloudy_mask


def computation_solo_properties(img, cloud_detector):
    data = compute_band_properties(img)
    data = dict(**data, **compute_unique_properties(img))
    data['cloud_percentage'] = cloud_detector.compute_percentage_cloud(img)
    return data


def find_scenes(path):
    path_type = path / "s2"
    for season in path_type.iterdir():
        roi, season_name = season.name.split('_')[:2]
        for scene in season.iterdir():
            yield (roi, season_name, int(scene.name.split('_')[-1])), find_patches(path, roi, season_name, scene)


def find_patches(path, roi, season_name, scene):
    for patch in scene.iterdir():
        scenario_id = int(scene.name.split('_')[-1])
        patch_id = int(os.path.splitext(patch.name)[0].split('_')[-1][1:])
        yield (Patch(roi, season_name, scenario_id, patch_id, "cloudless", patch),
               Patch(roi, season_name, scenario_id, patch_id, "cloudy",
                     path / "s2_cloudy" / f"{roi}_{season_name}_s2_cloudy" / f"s2_cloudy_{scenario_id}" / f"{roi}_{season_name}_s2_cloudy_{scenario_id}_p{patch_id}.tif"))
