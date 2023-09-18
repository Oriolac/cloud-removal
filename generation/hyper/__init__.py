import torch
from pytorch_msssim import SSIM
from torch import optim, nn
from torch.nn import BCELoss, MSELoss, L1Loss


class PSNR(nn.Module):

    def __init__(self, range=1):
        super(PSNR, self).__init__()
        self.mse = MSELoss()
        self.range = range

    def forward(self, x, y):
        den = self.mse(x, y)
        return 10 * torch.log10(self.range / den)


OPTIMIZERS = {
    "Adam": optim.Adam,
}

CRITERIONS = {
    "BCELoss": BCELoss,
    "MSELoss": MSELoss,
    "L1Loss": L1Loss,
    "SSIM": SSIM,
    "PSNR": PSNR,
}


def get_optimizer(model, optim):
    return OPTIMIZERS[optim.name](model.parameters(), **optim.kwargs)


def get_criterion(criterion):
    return CRITERIONS[criterion]()


def get_metric(name, kwargs):
    return CRITERIONS[name](**kwargs)


if __name__ == '__main__':
    t = torch.ones(1, 13, 256, 256)
    b = torch.zeros(1, 13, 256, 256) + 0.1
    print({"BCELoss": BCELoss()(t, b),
     "MSELoss": MSELoss()(t, b),
     "L1Loss": L1Loss()(t, b),
     "SSIM": SSIM(channel=13, data_range=10000)(t, b),
     "PSNR": PSNR()(t, b), })
