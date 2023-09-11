import logging

import numpy as np
import torch
from torch import nn

from cnn.config import Model, Train, Checkpoint
from utils import weights_init


class ResnetStackedArchitecture(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(ResnetStackedArchitecture, self).__init__()
        self.F = 256
        self.B = 16
        self.kernel_size = 3
        self.padding_size = 1
        self.scale_res = 0.1
        self.dropout = False
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.channels_distance = self.input_channels - self.output_channels

        model = [
            nn.Conv2d(self.input_channels, self.F, kernel_size=self.kernel_size, padding=self.padding_size, bias=True),
            nn.ReLU(True)]
        # generate a given number of blocks
        for i in range(self.B):
            model += [ResnetBlock(self.F, use_dropout=self.dropout, use_bias=True,
                                  res_scale=self.scale_res, padding_size=self.padding_size)]

        model += [
            nn.Conv2d(self.F, self.output_channels, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # long-skip connection: add cloudy MS input (excluding the trailing two SAR channels) and model output
        return input[:, self.channels_distance:, ...] + self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout, use_bias, res_scale=0.1, padding_size=1):
        super(ResnetBlock, self).__init__()
        self.res_scale = res_scale
        self.padding_size = padding_size
        self.conv_block = self.build_conv_block(dim, use_dropout, use_bias)

        # conv_block:
        #   CONV (pad, conv, norm),
        #   RELU (relu, dropout),
        #   CONV (pad, conv, norm)

    def build_conv_block(self, dim, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias)]
        conv_block += [nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # add residual mapping
        out = x + self.res_scale * self.conv_block(x)
        return out


def get_model(config: Train, device):
    name = config.model.name
    model = MODELS[name](**config.model.kwargs).to(device)
    model = model.apply(weights_init)
    return model


MODELS = {
    "DSen2-CR": ResnetStackedArchitecture
}

if __name__ == '__main__':
    net = ResnetStackedArchitecture(15, 13)
    t = torch.Tensor(1, 15, 256, 256)
    res = net(t)
    print(res.shape)
    num_model_parameters = sum([p.numel() for p in net.parameters()])
    print(num_model_parameters)
