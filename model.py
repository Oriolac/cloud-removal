import torch
import torch.nn as nn

discriminator = nn.Sequential(
    # inn: 3 x 224 x 224

    nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.2, inplace=True),
    # out : 32 x 128 x 128

    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 56 x 56

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 28 x 28

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out 256 x 14 x 14

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=2, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out 512 x 7 x 7

    nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=2, bias=False),
    nn.BatchNorm2d(1024),
    nn.LeakyReLU(0.2, inplace=True),
    # out 1024 x 4 x 4

    nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
)


class Comparator(nn.Module):

    def reduction_layer(self, input, output):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, inplace=True), )


generative = nn.Sequential(
    # inn: 3 x 224 x 224

    nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    # out 16 x 128 x 128

    nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    # out 32 x 56 x 56

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out 64 x 56 x 56

    nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    # out 32 x 56 x 56

    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    # out 16 x 128 x 128

    nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False),
    nn.Tanh()
)
