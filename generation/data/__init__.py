from torch.utils.data import DataLoader

from data.config import Data
from data.datasets import *
from pathlib import Path

DATASETS = {
    "S1_2_CR_Data": S1_2_CR_Data,
    "S2_CR_Data": S2_CR_Data,
    "S1_2_CR_Mask_Data": S1_2_CR_Mask_Data
}

SETS = ["train", "val", "test"]


def get_dataloaders(config: Data):
    Dataset = DATASETS[config.dataset]
    dloaders = {}
    for set, batch_size in zip(SETS, [config.batch_size, config.batch_size, config.batch_size]):
        path_set = Path(config.input) / "{set}.pkl".format(set=set)
        dataset = Dataset(path_set, config.root_imgs)
        shuffle = set == "train"
        dloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers)
        dloaders[set] = dloader
    return [dloaders[set] for set in SETS]
