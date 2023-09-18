import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm import tqdm

from data import get_dataloaders
from data.config import Data
from hyper import PSNR
from utils import calculate_psnr, calculate_ssim, calculate_carl, calculate_sam

IDXS = [5278, 4294, 316, 7264, 2729, 6317, 2266, 4065, 746, 3968,
        4287, 581, 1302, 5349, 4433, 7657, 5254, 6553, 4438, 6780]


def scaled(x):
    min_val, max_val = np.percentile(x, (2, 98))

    # Scale the pixel values to the range of 0-255
    return np.interp(x, (min_val, max_val), (0.01, 255)).astype(np.uint8)


def main(run, name, folder, gpu):
    data_kwargs = {
        "dataset": "S1_2_CR_Mask_Data",
        "root_imgs": "/data/",
        "input": "/inputs/augmented/",
        "batch_size": 1,
        "num_workers": 20
    }
    torch.manual_seed(42)
    device = torch.device("cuda:0") if gpu == "all" else torch.device(f"cuda:{gpu}")
    data = Data(**data_kwargs)
    train_loader, val_loader, test_loader = get_dataloaders(data)
    model = get_model(device, run)
    #model = ResnetStackedArchitecture(15, 256, 16, 13)
    #state_dict = torch.load(r"./baseline_resnet.pth")
    #model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    """
    ssims = []
    psnrs = []
    mae = 0
    carl = 0
    sam = 0
    calculate_mae = torch.nn.L1Loss(reduction='mean')
    for idx, img, target, mask in tqdm(test_loader):
        ssims_batch, psnsrs_batch = [], []
        x = img.to(device)
        y = target.to(device)
        mask = mask.to(device)
        y_hat = model(x)
        res_mae = calculate_mae(y_hat, y)
        carl += calculate_carl(x, y, y_hat, mask)
        mae += (res_mae.detach().item())
        sam += calculate_sam(y, y_hat).item()
        print(PSNR()(y, y_hat))
        for i in range(img.size(0)):

            ssims_batch.append(calculate_ssim(y[i], y_hat[i]))
            psnsrs_batch.append(calculate_psnr(y[i], y_hat[i]))
        ssims.append(float(np.mean(ssims_batch)))
        psnrs.append(float(np.mean(psnsrs_batch)))
    ssim = float(np.mean(ssims))
    psnr = float(np.mean(psnrs))
    mae = mae / len(test_loader.dataset)
    carl = carl / len(test_loader.dataset)
    sam = sam / len(test_loader.dataset)
    parameters = sum((p.numel() for p in model.parameters()))
    print(name, parameters, mae, ssim, psnr, carl, sam, sep=',')"""
    for idx in IDXS:
        _, img, target, mask = test_loader.dataset[idx]
        img = img.to(device)[np.newaxis, ...]
        output = model(img)
        array = output.detach().cpu().numpy()[0]
        save_array(array, folder, idx, "output")
        save_array(mask.detach().cpu().numpy()[np.newaxis, ...], folder, idx, "mask")
        save_array(img.detach().cpu().numpy()[0], folder, idx, "input")
        save_array(target.detach().cpu().numpy(), folder, idx, "target")


def get_model(device, run):
    artifact_path = pathlib.Path(f"/runs/112291678684477224/{run}/artifacts/models/")
    artifact_path = next(artifact_path.iterdir()) / "data" / "model.pth"
    model = torch.load(artifact_path.absolute(), map_location=device)
    return model


def save_array(array, folder, idx, type):
    with rasterio.open(f"{folder}/{idx}_{type}.tif", 'w', driver='GTiff',
                       height=array.shape[1],
                       width=array.shape[2],
                       count=array.shape[0],
                       dtype=str(array.dtype)) as dst:
        for i in range(array.shape[0]):
            dst.write(array[i], i + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=str)
    parser.add_argument("name", type=str)
    parser.add_argument("folder", type=str)
    parser.add_argument("gpu", type=int)
    main(**parser.parse_args().__dict__)
