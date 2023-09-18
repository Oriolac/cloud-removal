import cnn.models as models
import torch.nn as nn
import pandas as pd
import time
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use("seaborn-v0_8")

def params(net):
    return sum([p.numel() for p in net.parameters()])

def time_computation(net, input):
    start = time.time()
    net(input)
    return time.time() - start


def get_params(filters, step=1, out=lambda x: x):
    res = defaultdict(list)
    filters = list(range(1, filters, step))
    for f in tqdm(filters):
        conv = nn.Conv2d(f, out(f), kernel_size=(3,3)).to("cuda:1")
        seq = nn.Sequential(nn.Conv2d(f, out(f), kernel_size=(3,3), bias=False), nn.BatchNorm2d(out(f)), nn.Conv2d(out(f), out(f), kernel_size=(3,3), bias=False), nn.BatchNorm2d(out(f))).to("cuda:1")
        resb = models.ResidualBlock(f, out(f)).to("cuda:1")
        for i in range(250):
            res["Filters"].append(f)
            t = torch.rand(1, f, 128, 128).to("cuda:1")
            res["Convolutional params"].append(params(conv))
            res["Convolutional time"].append(time_computation(conv, t))
            res["Sequential params"].append(params(seq))
            res["Sequential time"].append(time_computation(seq, t))
            res["Residual params"].append(params(resb))
            res["Residual time"].append(time_computation(resb, t))
    return pd.DataFrame(res).groupby(by=["Filters"]).mean()

df = get_params(256, step=2, out=lambda x: x)
df.to_csv('/data/output/cnn_layers_comparison.csv')

df = get_params(256, step=2, out=lambda x: x * 2)
df.to_csv('/data/stats/cnn_layers_comparison2.csv')
