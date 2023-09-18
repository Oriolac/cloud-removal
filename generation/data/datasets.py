import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio as rio
from PIL import Image

clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
clip_max = [[0, 0], [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
            [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]


def get_sar_processed(data_image):
    data_type = 1
    max_val = 2

    for channel in range(2):
        data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                      clip_max[data_type - 1][channel])
        data_image[channel] = data_image[channel].astype(float) - clip_min[data_type - 1][channel]
        data_image[channel] = max_val * (data_image[channel] / (
                clip_max[data_type - 1][channel] - clip_min[data_type - 1][channel]))
    return data_image


def get_s2_processed(data_image):
    data_type = 3
    for channel in range(len(data_image)):
        data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                      clip_max[data_type - 1][channel])
    data_image = data_image / 2000
    return data_image

def open_image(path, i=1, s=13):
    img = rio.open(path)
    img = np.stack([img.read(b) for b in range(i, s + 1)])
    #img = get_sar_processed(img)
    #img[2:] = get_s2_processed(img[2:])
    return img


class S1_2_CR_Mask_Data(Dataset):
    def __init__(self, pkl_file: str, root_dir: str):
        self.df = pd.read_pickle(pkl_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def get_path(self, path_col):
        return os.path.join(self.root_dir, path_col)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cloudy_path = self.get_path(row['cloudly_img_path'])
        cloudless_path = self.get_path(row['cloudless_img_path'])
        sar_path = self.get_path(row['sar_img_path'])
        mask_cloud = self.get_path("masks/" + row['cloud_mask_path'] + ".tiff")
        mask_cloud = np.array(Image.open(mask_cloud))
        cloudy = open_image(cloudy_path)
        label = open_image(cloudless_path)
        sar = open_image(sar_path, 1, 2)
        img = torch.from_numpy(np.concatenate([sar, cloudy]).astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        return idx, img, label, torch.from_numpy(mask_cloud.astype(np.float32))


class S1_2_CR_Data(Dataset):

    def __init__(self, pkl_file: str, root_dir: str):
        self.df = pd.read_pickle(pkl_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def get_path(self, path_col):
        return os.path.join(self.root_dir, path_col)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cloudy_path = self.get_path(row['cloudly_img_path'])
        cloudless_path = self.get_path(row['cloudless_img_path'])
        sar_path = self.get_path(row['sar_img_path'])
        cloudy = open_image(cloudy_path)
        label = open_image(cloudless_path)
        sar = open_image(sar_path, 1, 2)
        img = torch.from_numpy(np.concatenate([sar, cloudy]).astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        return img, label


class S2_CR_Data(Dataset):

    def __init__(self, pkl_file: str, root_dir: str):
        self.df = pd.read_pickle(pkl_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def get_path(self, path_col):
        return os.path.join(self.root_dir, path_col)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cloudy_path = self.get_path(row['cloudly_img_path'])
        cloudless_path = self.get_path(row['cloudless_img_path'])
        cloudy = open_image(cloudy_path)
        label = open_image(cloudless_path)
        img = torch.from_numpy(cloudy.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        return img, label
