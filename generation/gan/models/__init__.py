import torch
from torch import nn

from cnn.models import ResidualCNN
from data.config import Model
from utils import weights_init


class Daniel(nn.Module):
    def __init__(self, input_channel, hidden_channels, relu):
        super(Daniel, self).__init__()
        self.input_channel = input_channel
        last_channel = hidden_channels[0]
        mods = []
        for hidden_ch in hidden_channels:
            mods.append(self.conv_disc_block(last_channel, hidden_ch, relu))
            last_channel = hidden_ch

        self.model = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channels[0], kernel_size=(4, 4), stride=(2, 2), padding=1),
            # Assuming output_channels of the generator is 13
            nn.LeakyReLU(relu),
            *mods,
            nn.Conv2d(last_channel, 1, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def conv_disc_block(self, in_, out, relu):
        return nn.Sequential(nn.Conv2d(in_, out, kernel_size=(4, 4), stride=(2, 2), padding=1),
                             nn.BatchNorm2d(out),
                             nn.LeakyReLU(relu))

    def forward(self, img):
        return self.model(img).reshape(-1)


class Didac(nn.Module):
    def __init__(self, input_channel):
        super(Didac, self).__init__()
        self.input_channel = input_channel
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            # Assuming output_channels of the generator is 13
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).reshape(-1)


def get_model(config_model: Model, device):
    name = config_model.name
    model = MODELS[name](**config_model.kwargs).to(device)
    model = model.apply(weights_init)
    return model


MODELS = {
    "Residual-CNN": ResidualCNN,
    "Didac": Didac,
    "Daniel": Daniel
}

if __name__ == '__main__':
    t = torch.Tensor(15, 13, 256, 256)
    net = Didac(13)
    print(net(t).shape)
