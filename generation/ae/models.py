import torch
from torch import nn

from ae.config import Train
from cnn.models import ResidualBlock
from utils import weights_init



class AusiasConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(AusiasConv, self).__init__()

        # Contracting/Downsampling Path
        self.enc1 = self.conv_block(input_channel, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expanding/Upsampling Path
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.out_conv = nn.Sequential(nn.Conv2d(64, output_channel, kernel_size=(1, 1)), nn.Sigmoid())

    def forward(self, x):
        # Contracting/Downsampling Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        # Expanding/Upsampling Path
        dec3 = self.dec3(torch.cat([enc3, self.upconv3(enc4)], dim=1))
        dec2 = self.dec2(torch.cat([enc2, self.upconv2(dec3)], dim=1))
        dec1 = self.dec1(torch.cat([enc1, self.upconv1(dec2)], dim=1))

        return self.out_conv(dec1) * 10000

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
        )


class AusiasBatch(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(AusiasBatch, self).__init__()

        # Contracting/Downsampling Path
        self.enc1 = self.conv_block(input_channel, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expanding/Upsampling Path
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, output_channel, kernel_size=(1, 1))

    def forward(self, x):
        # Contracting/Downsampling Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        # Expanding/Upsampling Path
        dec3 = self.dec3(torch.cat([enc3, self.upconv3(enc4)], dim=1))
        dec2 = self.dec2(torch.cat([enc2, self.upconv2(dec3)], dim=1))
        dec1 = self.dec1(torch.cat([enc1, self.upconv1(dec2)], dim=1))

        return self.out_conv(dec1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )


class Ausias(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Ausias, self).__init__()

        # Contracting/Downsampling Path
        self.enc1 = self.conv_block(input_channel, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expanding/Upsampling Path
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.out_conv = nn.Sequential(nn.Conv2d(64, output_channel, kernel_size=(1, 1)), nn.Sigmoid())

    def forward(self, x):
        # Contracting/Downsampling Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        # Expanding/Upsampling Path
        dec3 = self.dec3(torch.cat([enc3, self.upconv3(enc4)], dim=1))
        dec2 = self.dec2(torch.cat([enc2, self.upconv2(dec3)], dim=1))
        dec1 = self.dec1(torch.cat([enc1, self.upconv1(dec2)], dim=1))

        return self.out_conv(dec1) * 10000

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )


class Anna(nn.Module):

    def __init__(self, input_channel, hidden_channels, output_channel):
        super(Anna, self).__init__()
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels
        self.output_channel = output_channel
        last_channel = input_channel
        for hidden_channel in hidden_channels:
            setattr(self, "convblock_{}".format(hidden_channel),
                    self.conv_block(last_channel, hidden_channel))
            last_channel = hidden_channel
        for hidden_channel in reversed(hidden_channels[:-1]):
            setattr(self, "deconvblock_{}".format(hidden_channel),
                    self.deconv_block(last_channel, hidden_channel))
            last_channel = hidden_channel
        setattr(self, "deconvblock_{}".format(output_channel),
                self.deconv_block(last_channel, output_channel))
        self.final = nn.Sequential(nn.Sigmoid())

    def conv_block(self, input_filters, output_filters):
        return nn.Sequential(nn.Conv2d(input_filters, output_filters, kernel_size=(3, 3,), stride=(2, 2), padding=1),
                             nn.BatchNorm2d(output_filters), nn.ReLU(inplace=True)
                             )

    def deconv_block(self, input_filters, output_filters):
        return nn.Sequential(
            nn.ConvTranspose2d(input_filters, output_filters, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.Conv2d(output_filters, output_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        for channel in self.hidden_channels:
            conv = getattr(self, "convblock_{}".format(channel))
            x = conv(x)
        for channel in reversed(self.hidden_channels[:-1]):
            deconv = getattr(self, "deconvblock_{}".format(channel))
            x = deconv(x)
        deconv = getattr(self, "deconvblock_{}".format(self.output_channel))
        x = deconv(x)
        return self.final(x) * 10000


class AnnaSkip(nn.Module):

    def __init__(self, input_channel, hidden_channels, output_channel):
        super(AnnaSkip, self).__init__()
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels
        self.output_channel = output_channel
        last_channel = input_channel
        for hidden_channel in hidden_channels:
            setattr(self, "convblock_{}".format(hidden_channel),
                    self.conv_block(last_channel, hidden_channel))
            last_channel = hidden_channel
        for hidden_channel in reversed(hidden_channels[:-1]):
            setattr(self, "deconvblock_{}".format(hidden_channel),
                    self.deconv_block(last_channel, hidden_channel))
            last_channel = hidden_channel
        setattr(self, "deconvblock_{}".format(output_channel),
                self.deconv_block(last_channel, output_channel))
        self.final = nn.Sigmoid()

    def conv_block(self, last_channel, output_channel):
        return nn.Sequential(nn.Conv2d(last_channel, output_channel, kernel_size=(3, 3,), stride=(2, 2), padding=1),
                             nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True)
                             )

    def deconv_block(self, input_filters, output_filters):
        return nn.Sequential(
            nn.ConvTranspose2d(input_filters, output_filters, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.Conv2d(output_filters, output_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        connections = []
        for channel in self.hidden_channels:
            conv = getattr(self, "convblock_{}".format(channel))
            x = conv(x)
            connections.insert(0, x)
        for connection, channel in zip(connections[1:], reversed(self.hidden_channels[:-1])):
            deconv = getattr(self, "deconvblock_{}".format(channel))
            x = deconv(x)
            x += connection
        deconv = getattr(self, "deconvblock_{}".format(self.output_channel))
        x = deconv(x)
        return self.final(x) * 10000


def test_anna():
    net = Anna(15, [64, 128, 256], 13)
    t = torch.Tensor(1, 15, 256, 256)
    output = net(t)
    print(output.shape)
    num_model_parameters = sum([p.numel() for p in net.parameters()])
    print(num_model_parameters)


def test_anna_skip():
    net = AnnaSkip(15, [64, 128, 256], 13)
    t = torch.Tensor(1, 15, 256, 256)
    output = net(t)
    print(output.shape)
    num_model_parameters = sum([p.numel() for p in net.parameters()])
    print(num_model_parameters)


def test_ausias():
    net = Ausias(15, 13)
    t = torch.Tensor(1, 15, 256, 256)
    output = net(t)
    print(output.shape)
    num_model_parameters = sum([p.numel() for p in net.parameters()])
    print(num_model_parameters)


def get_model(config: Train, device):
    name = config.model.name
    model = MODELS[name](**config.model.kwargs).to(device)
    model = model.apply(weights_init)
    return model


MODELS = {
    "Anna": Anna,
    "AnnaSkip": AnnaSkip,
    "Ausias": Ausias,
    "AusiasBatch": AusiasBatch,
    "AusiasConv": AusiasConv
}

if __name__ == '__main__':
    test_anna()
    test_anna_skip()
    test_ausias()
