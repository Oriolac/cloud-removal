import argparse

import numpy as np
import pandas as pd
from s2cloudless import S2PixelCloudDetector
import rasterio as rio
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_detect_cloud_bands(img):
    bands = [img.read(b) / 10000 for b in img.indexes]
    bands = np.stack(bands)
    return bands.transpose(1,2,0)

def main(input, root, output):
    cloud_detector = S2PixelCloudDetector(threshold=0.7, average_over=4, dilation_size=2, all_bands=True)
    path_input = Path(input)
    for split in ["train.pkl", "val.pkl", "test.pkl"]:
        df = pd.read_pickle(path_input / split)
        cloud_mask_paths = []
        for i, row in tqdm(df.iterrows()):
            scene, season, roi, patch = i
            img = rio.open("{}/{}".format(root, row['cloudly_img_path']))
            img = get_detect_cloud_bands(img)
            mask = cloud_detector.get_cloud_masks(img[np.newaxis, ...])[0]
            mask_path = "{}_{}_{}_{}.tif".format(roi,  scene, season, patch)
            img = Image.fromarray(mask)
            img.save("{}/{}.tiff".format(output, mask_path))
            cloud_mask_paths.append(mask_path)
        df["cloud_mask_path"] = cloud_mask_paths
        df.to_pickle(path_input / split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("root", type=str)
    parser.add_argument("output", type=str)
    main(**parser.parse_args().__dict__)

