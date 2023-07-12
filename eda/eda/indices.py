import numpy as np
from numba import jit

from .process import process_bands


def extract_data_indexes(img):
    img = process_bands(img, lambda x: x)
    threshold = np.vectorize(lambda x: x + 0.00001)
    data = {}
    ndvi = (img[:, :, 7] - img[:, :, 3]) / threshold(img[:, :, 7] + img[:, :, 3])
    data = dict(**data, **get_index_properties('ndvi', ndvi))
    gndvi = (img[:, :, 7] - img[:, :, 2]) / threshold(img[:, :, 7] + img[:, :, 2])
    data = dict(**data, **get_index_properties('gndvi', gndvi))
    ndmi = (img[:, :, 7] - img[:, :, 11]) / threshold(img[:, :, 7] + img[:, :, 11])
    data = dict(**data, **get_index_properties('ndmi', ndmi))
    msi = (img[:, :, 11]) / threshold(img[:, :, 7])
    data = dict(**data, **get_index_properties('msi', msi))
    nbri = (img[:, :, 7] - img[:, :, 12]) / threshold(img[:, :, 7] + img[:, :, 12])
    data = dict(**data, **get_index_properties('nbri', nbri))
    bsi = (img[:, :, 11] + img[:, :, 3] - img[:, :, 7] - img[:, :, 1]) / threshold(
        img[:, :, 11] + img[:, :, 3] + img[:, :, 7] + img[:, :, 1])
    data = dict(**data, **get_index_properties('bsi', bsi))
    ndwi = (img[:, :, 2] - img[:, :, 7]) / threshold(img[:, :, 2] + img[:, :, 7])
    data = dict(**data, **get_index_properties('ndwi', ndwi))
    ndsi = (img[:, :, 2] - img[:, :, 11]) / threshold(img[:, :, 2] + img[:, :, 11])
    data = dict(**data, **get_index_properties('ndsi', ndsi))
    return data

@jit
def get_index_properties(name, index):
    data = {}
    data[f'{name}_mean'] = index.mean()
    data[f'{name}_median'] = np.percentile(index, 50)
    data[f'{name}_max'] = index.max()
    data[f'{name}_min'] = index.min()
    data[f'{name}_std'] = index.std()
    data[f'{name}_p25'] = np.percentile(index, 25)
    data[f'{name}_under25'] = (index < data[f'{name}_p25']).sum()
    data[f'{name}_p75'] = np.percentile(index, 75)
    data[f'{name}_over75'] = (index > data[f'{name}_p75']).sum()
    return data
