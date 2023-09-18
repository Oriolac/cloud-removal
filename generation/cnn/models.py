import logging

import numpy as np
import torch
from torch import nn

from cnn.config import Model, Train, Checkpoint
from utils import weights_init


class Carla(nn.Module):

    def __init__(self, input, chan1, chan2, chan3):
        super(Carla, self).__init__()
        self.model = nn.Sequential(self.conv(input, chan1),  self.conv(chan1, chan2), self.conv(chan2, chan3), self.conv(chan3, 13), )
        self.final = nn.Sigmoid()

    def conv(self, in_, out_):
        return nn.Sequential(nn.Conv2d(in_, out_, (3, 3), padding=(1, 1)), nn.BatchNorm2d(out_), nn.ReLU(), )

    def forward(self, x):
        return self.final(self.model(x) + x[:, 2:]) * 10000


class ResidualSkipCNN(nn.Module):

    def __init__(self, input_channels, hidden_channels, output_channels):
        super(ResidualSkipCNN, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.seq1 = []
        last_channel = input_channels
        for hidden_channel in hidden_channels:
            self.seq1.append(ResidualBlock(last_channel, hidden_channel))
            last_channel = hidden_channel
        self.seq1 = nn.Sequential(*self.seq1)
        self.seq2 = []
        for hidden_channel in reversed(hidden_channels):
            self.seq2.append(ResidualBlock(last_channel, hidden_channel))
            last_channel = hidden_channel
        self.seq2 = nn.Sequential(*self.seq2)
        self.final = nn.Sequential(
            nn.Conv2d(last_channel, self.output_channels, kernel_size=(3, 3), padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.seq1(x)
        z = self.seq2(x)
        return self.final(z) * 10000

class ResidualCNN(nn.Module):

    def __init__(self, input_channels, hidden_channels, output_channels):
        super(ResidualCNN, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.seq1 = []
        last_channel = input_channels
        for hidden_channel in hidden_channels:
            self.seq1.append(ResidualBlock(last_channel, hidden_channel))
            last_channel = hidden_channel
        self.seq1 = nn.Sequential(*self.seq1)
        self.seq2 = []
        for hidden_channel in reversed(hidden_channels):
            self.seq2.append(ResidualBlock(last_channel, hidden_channel))
            last_channel = hidden_channel
        self.seq2 = nn.Sequential(*self.seq2)
        self.final = nn.Sequential(
            nn.Conv2d(last_channel, self.output_channels, kernel_size=(3, 3), padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        return self.final(x) * 10000


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResnetStackedArchitecture(nn.Module):

    def __init__(self, input_channels, F, B, output_channels):
        super(ResnetStackedArchitecture, self).__init__()
        self.F = F
        self.B = B
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
            nn.Conv2d(self.F, 64, kernel_size=self.kernel_size, padding=self.padding_size, bias=True), # model.18
            nn.Dropout(),
            nn.Conv2d(64, 13, kernel_size=self.kernel_size, padding=self.padding_size, bias=True),  # model.19
        ]

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
    "DSen2-CR": ResnetStackedArchitecture,
    "Residual-CNN": ResidualCNN,
    "Carla": Carla
}


def resnet_stacked_test():
    net = ResnetStackedArchitecture(15, 256, 16, 13)
    state_dict = torch.load(r"./baseline_resnet.pth")
    print(state_dict.keys())
    net_state_dict = net.state_dict()
    print(net_state_dict.keys())
    print(state_dict['model.18.weight'].shape)
    print(state_dict['model.20.weight'].shape)
    for (name, value) in state_dict.items():
        print(name, value.shape == net_state_dict[name].shape)
    #state_dict['model.19.weight'] = state_dict['model.20.weight']
    #state_dict['model.19.bias'] = state_dict['model.20.bias']
    #state_dict.pop('model.20.weight')
    #state_dict.pop('model.20.bias')
    print(state_dict.keys())
    net.load_state_dict(state_dict)
    t = torch.Tensor(1, 15, 256, 256)
    res = net(t)
    print(res.shape)
    num_model_parameters = sum([p.numel() for p in net.parameters()])
    print(num_model_parameters)
    rblock = ResidualBlock(3, 3)


def test_resnet_blocks():
    kwargs = {
        "input_channels": 15,
        "output_channels": 13,
        "hidden_channels": [64, 128, 256],
    }
    net = ResidualCNN(**kwargs)
    t = torch.Tensor(1, 15, 256, 256)
    res = net(t)
    print("Output shape: {}".format(res.shape))
    num_model_parameters = sum([p.numel() for p in net.parameters()])
    print("Model parameters: {}".format(num_model_parameters))


def test_carla():
    net = Carla(15, 64, 128, 256)
    t = torch.Tensor(1, 15, 256, 256)
    res = net(t)
    print("Output shape: {}".format(res.shape))
    num_model_parameters = sum([p.numel() for p in net.parameters()])
    print("Model parameters: {}".format(num_model_parameters))

if __name__ == '__main__':
    resnet_stacked_test()
    #test_resnet_blocks()
    #test_carla()