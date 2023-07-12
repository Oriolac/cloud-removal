import dataclasses
import datetime as dt
import os
import pathlib

import rasterio as rio

from .bands import compute_band_properties, compute_paired_mse_bands, compute_kde_patch
from .cloud import CloudDetector, get_mask_crop_correlation
from .indices import extract_data_indexes
from .process import get_detect_cloud_bands
from .unique import compute_unique_properties


@dataclasses.dataclass
class Patch_CR:
    roi: str
    season: str
    scenario: int
    patch: int
    type: str
    path: pathlib.Path


@dataclasses.dataclass
class Patch_TS:
    roi: str
    season: str
    scenario: int
    patch: int
    time_interval: int
    time: dt.datetime
    path: pathlib.Path


def add_patch_cr_data(data, patch: Patch_CR):
    data['roi'] = patch.roi
    data['season'] = patch.season
    data['patch'] = patch.patch
    data['scene'] = patch.scenario


def add_patch_ts_data(data, patch: Patch_TS):
    data['roi'] = patch.roi
    data['season'] = patch.season
    data['patch'] = patch.patch
    data['scene'] = patch.scenario
    data['time_interval'] = patch.time_interval
    data['time'] = patch.time


def compute_all_properties_cr(cloudless: Patch_CR, cloudy: Patch_CR):
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

    add_patch_cr_data(data_cloudless, cloudless)
    add_patch_cr_data(data_cloudy, cloudy)
    add_patch_cr_data(data_paired, cloudless)

    return data_cloudless, data_cloudy, data_paired, cloudless_kde, cloudy_kde, cloudy_mask


def compute_cloud_percentage(patch: Patch_TS):
    cloud_detector = CloudDetector()
    img = rio.open(patch.path)
    data = computation_solo_properties(img, cloud_detector)
    add_patch_ts_data(data, patch)
    return data

def computation_solo_properties(img, cloud_detector):
    data = compute_band_properties(img)
    data = dict(**data, **compute_unique_properties(img))
    data['cloud_percentage'] = cloud_detector.compute_percentage_cloud(img)
    data_indices = extract_data_indexes(img)
    data = dict(**data, **data_indices)
    return data


def find_scenes_cr(path):
    path_type = path / "s2"
    for season in path_type.iterdir():
        roi, season_name = season.name.split('_')[:2]
        for scene in season.iterdir():
            yield (roi, season_name, int(scene.name.split('_')[-1])), find_patches_cr(path, roi, season_name, scene)


def find_scenes_ts(path):
    for path_roi in path.iterdir():
        for path_scene in path_roi.iterdir():
            path_type = path_scene / "S2"
            for path_time_inteval in path_type.iterdir():
                roi = path_roi.name
                scene = path_scene.name
                interval = path_time_inteval.name
                yield ((roi, scene, interval), find_patches_ts(path_time_inteval, roi, scene, interval))


def get_season_from_ts_datetime(datetime_obj):
    seasons_datetimes = [
        ('winter', (dt.datetime(datetime_obj.year, 1, 1), dt.datetime(datetime_obj.year, 3, 20))),
        ('spring', (dt.datetime(datetime_obj.year, 3, 21), dt.datetime(datetime_obj.year, 6, 20))),
        ('summer', (dt.datetime(datetime_obj.year, 6, 21), dt.datetime(datetime_obj.year, 9, 22))),
        ('autumn', (dt.datetime(datetime_obj.year, 9, 23), dt.datetime(datetime_obj.year, 12, 20))),
        ('winter', (dt.datetime(datetime_obj.year, 12, 21), dt.datetime(datetime_obj.year, 12, 31)))
    ]
    for season_name, (start_datetime, end_datetime) in seasons_datetimes:
        if start_datetime <= datetime_obj <= end_datetime:
            return season_name
    raise NameError("Impossible! Not finding the season :(")


def find_patches_ts(path, roi, scene, interval_time):
    for patch_path in path.iterdir():
        year, month, day = list(map(int, patch_path.name.split('_')[5].split('-')))
        date_patch = dt.datetime(year, month, day)
        patch_id = int(os.path.splitext(patch_path.name)[0].split('_')[-1])
        yield Patch_TS(roi, get_season_from_ts_datetime(date_patch), int(scene), patch_id, int(interval_time),
                       date_patch,
                       patch_path)


def find_patches_cr(path, roi, season_name, scene):
    for patch in scene.iterdir():
        scenario_id = int(scene.name.split('_')[-1])
        patch_id = int(os.path.splitext(patch.name)[0].split('_')[-1][1:])
        yield (Patch_CR(roi, season_name, scenario_id, patch_id, "cloudless", patch),
               Patch_CR(roi, season_name, scenario_id, patch_id, "cloudy",
                        path / "s2_cloudy" / f"{roi}_{season_name}_s2_cloudy" / f"s2_cloudy_{scenario_id}" / f"{roi}_{season_name}_s2_cloudy_{scenario_id}_p{patch_id}.tif"))
