import torch
from torch import nn

from cnn.config import Train
from cnn.models import ResidualBlock
from utils import weights_init


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class Print(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        print(x.shape)
        return x

class Vanessa(nn.Module):
    def __init__(self, input_channel, hidden_channels, output_channel, latent_dim):
        super(Vanessa, self).__init__()
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels
        self.output_channel = output_channel
        last_channel = input_channel
        for hidden_channel in hidden_channels:
            setattr(self, "convblock_{}_{}".format(last_channel, hidden_channel),
                    self.conv_block(last_channel, hidden_channel))
            last_channel = hidden_channel

        self.lineal = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(last_channel,
                                              latent_dim * 2))
        self.deslineal = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 8 * 8))
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 1024, (3,3), padding=1),
            nn.BatchNorm2d(1024),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 1024, (3, 3), padding=1),
            nn.BatchNorm2d(1024),
        )

        for hidden_channel in reversed(hidden_channels[:-1]):
            setattr(self, "deconvblock_{}_{}".format(last_channel, hidden_channel),
                    self.deconv_block(last_channel, hidden_channel))
            last_channel = hidden_channel
        setattr(self, "deconvblock_{}".format(output_channel),
                self.deconv_block(last_channel, output_channel))
        self.final = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2,), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
            conv = getattr(self, "convblock_{}_{}".format(x.size(1), channel))
            x = conv(x)
            connections.insert(0, x)
        x = self.lineal(x)
        # LINEAL TO hidden
        mu, log_var = torch.chunk(x, 2, dim=-1)
        z = reparameterize(mu, log_var)
        # DIM TO OUTPUT
        z = self.deslineal(z)

        z = z.reshape(z.size(0), 1024, 8, 8)
        z = self.upsample(z)
        x = z + connections[0]
        for connection, channel in zip(connections[1:], reversed(self.hidden_channels[:-1])):
            deconv = getattr(self, "deconvblock_{}_{}".format(x.size(1),channel))
            x = deconv(x)
            x += connection
        conv = getattr(self, "deconvblock_{}".format(self.output_channel))
        x = conv(x)
        x = self.final(x)
        return x, mu, log_var


def get_model(config: Train, device):
    name = config.model.name
    model = MODELS[name](**config.model.kwargs).to(device)
    model = model.apply(weights_init)
    return model


MODELS = {
    "Vanessa": Vanessa
}


def test_vae():
    net = Vanessa(15, [128, 512, 1024], 13, 13)
    t = torch.Tensor(1, 15, 256, 256)
    output, mu, log_var = net(t)
    print(output.shape, mu.shape, log_var.shape)



if __name__ == '__main__':
    test_vae()
