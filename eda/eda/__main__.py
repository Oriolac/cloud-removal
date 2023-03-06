import argparse
import multiprocessing
import os
import pathlib as pl
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from .io import map_func, find_scenes


def rms_contrast(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grey.std()


def michelson_contrast(img):
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]
    min = np.min(Y)
    max = np.max(Y)
    return (max - min) / (max + min)


def get_attrs(patch):
    roi, season, _, scene, patch_name = patch.name.split('.')[0].split('_')
    return roi, season, scene, patch_name


def scaled(x):
    min_val, max_val = np.percentile(x, (2, 98))

    # Scale the pixel values to the range of 0-255
    return np.interp(x, (min_val, max_val), (0, 255)).astype(np.uint8)


def sigmoid(x):
    return 255 / (1 + np.exp(-3 * (scaled(x) / 255 - 0.5)))


def save_row(_dict, row):
    for name, value in row.items():
        _dict[name].append(value)
    return _dict


def save_pkl(dict, output, roi, season_name, scene, keyword):
    pd.DataFrame(dict).to_pickle(os.path.join(output, f"{roi}_{season_name}_{scene}_{keyword}.pkl"))

def save_kde(dict_cloudless, dict_cloudy, output, roi, season_name, scene):
    for b, (y_less, y_dy) in enumerate(zip(dict_cloudless.values(), dict_cloudy.values())):
        fig, axs = plt.subplots(3, 1, figsize=(14, 10))
        axs[0].set_title(f"cloudless scene {scene} in {season_name}: band {b}")
        axs[1].set_title(f"cloudy scene {scene} in {season_name}: band {b}")
        axs[2].set_title(f"paired scene {scene} in {season_name}: band {b}")
        for line in y_less:
            axs[0].plot(np.linspace(0, 255, 80), line, alpha=0.01, c='r')
        for line in y_dy:
            axs[1].plot(np.linspace(0, 255, 80), line, alpha=0.01, c='b')
        axs[2].plot(np.linspace(0, 255, 80), np.array(y_less).mean(0), alpha=1, c='r', label="Cloudless")
        axs[2].plot(np.linspace(0, 255, 80), np.array(y_dy).mean(0), alpha=1, c='b', label="Cloudy")
        axs[2].legend()
        plt.savefig(os.path.join(output, f"{roi}_{season_name}_{scene}_{b}.png"))
        plt.close()

def save_mask(masks, output, roi, season_name, scene):
    length = len(masks)
    if length <= 0:
        return
    masks = np.array(masks).mean(0)
    fig, ax = plt.subplots()
    plt.title(f"Cloud coverage of {scene} ({length})")
    im = ax.imshow(masks, vmax=1, vmin=0)
    cbar = ax.figure.colorbar(im)
    cbar.outline.set_visible(False)
    cbar.ax.plot([0, 1], [3, 3], 'k', label="Ref")
    plt.savefig(os.path.join(output, f"mask_{roi}_{season_name}_{scene}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="output")
    parser.add_argument("--threads", default=64, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    path = pl.Path(args.input)
    _types = ["s2", "s2_cloudy"]
    all_masks = []
    for ((roi, season_name, scene), find_patches) in tqdm(find_scenes(path)):
        print(roi, season_name, scene)
        cloudless_dict = defaultdict(list)
        cloudy_dict = defaultdict(list)
        paired_dict = defaultdict(list)
        cloudless_kdes_dict = defaultdict(list)
        cloudy_kdes_dict = defaultdict(list)
        masks = []
        with multiprocessing.Pool(args.threads) as pool:
            properties_scenes = pool.starmap(map_func, find_patches)
        for properties_scene in properties_scenes:
            data_cloudless, data_cloudy, data_paired, cloudless_kde, cloudy_kde, cloudy_mask = properties_scene
            save_row(cloudless_dict, data_cloudless)
            save_row(cloudy_dict, data_cloudy)
            save_row(paired_dict, data_paired)
            save_row(cloudless_kdes_dict, cloudless_kde)
            save_row(cloudy_kdes_dict, cloudy_kde)
            masks.append(cloudy_mask)

        save_pkl(cloudless_dict, args.output, roi, season_name, scene, "cloudless")
        save_pkl(cloudy_dict, args.output, roi, season_name, scene, "cloudy")
        save_pkl(paired_dict, args.output, roi, season_name, scene, "paired")
        save_kde(cloudless_kdes_dict, cloudy_kdes_dict, args.output, roi, season_name, scene)
        save_mask(masks, args.output, roi, season_name, scene)
        all_masks.extend(masks)
    save_mask(all_masks, args.output, "all", "all", "all")


if __name__ == '__main__':
    main()
