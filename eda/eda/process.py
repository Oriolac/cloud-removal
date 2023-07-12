import numpy as np

def scaled(x):
    min_val, max_val = np.percentile(x, (2, 98))

    # Scale the pixel values to the range of 0-255
    return np.interp(x, (min_val, max_val), (0.0001, 255)).astype(np.uint8)

def sigmoid(x):
    return 255 / (1 + np.exp(-4 * (scaled(x) / 255 - 0.5)))


def process_bands(img, scaled=scaled):
    bands = [scaled(img.read(b)) for b in img.indexes]
    bands = np.stack(bands)
    return bands.transpose(1,2,0).astype(np.uint8)

def tci_names():
    return ["r", "g", "b"]

def get_tci(img):
    r = scaled(img.read(4))
    g = scaled(img.read(3))
    b = scaled(img.read(2))
    rgb = np.stack([r, g, b])
    rgb = (rgb.transpose(1, 2, 0)).astype(np.uint8)
    return rgb

def get_detect_cloud_bands(img):
    bands = [img.read(b) / 10000 for b in img.indexes]
    bands = np.stack(bands)
    return bands.transpose(1,2,0)
