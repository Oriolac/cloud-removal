import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
import torch.utils.tensorboard as tboard
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from functools import reduce
import torchvision.transforms as T

from model import discriminator, generative, Comparator

flatten = lambda z: reduce(lambda x, y: list(x) + list(y), z)

sigmoid = nn.Sigmoid()

comparator = Comparator()


def train_comparator(cloud, clear, opt_d, loss):
    opt_d.zero_grad()

    real_preds = comparator(torch.Tensor(clear, clear))
    real_targets = torch.ones(clear.size(0), 1)
    real_loss = loss(sigmoid(real_preds), real_targets)
    real_score = torch.mean(real_preds).item()

    fake_clear = generative(cloud)
    fake_preds = comparator(torch.Tensor(fake_clear, clear))
    fake_targets = torch.zeros(fake_clear.size(0), 1)
    fake_loss = loss(sigmoid(fake_preds), fake_targets)
    fake_score = torch.mean(fake_preds).item()

    loss = real_loss + fake_loss
    disc_loss = loss.item()
    loss.backward()
    opt_d.step()
    return disc_loss, real_score, fake_score


def train_generator_comparison(cloud, clear, opt_g, loss):
    opt_g.zero_grad()

    fake_imgs = generative(cloud)
    disc_preds = comparator(torch.Tensor(fake_imgs, clear))
    fake_targets = torch.ones(cloud.size(0), 1)
    loss = loss(disc_preds, fake_targets)
    gen_loss = loss.item()
    loss.backward()
    opt_g.step()
    return gen_loss


def train_discriminator(cloud, clear, opt_d, loss):
    opt_d.zero_grad()

    real_preds = discriminator(clear)
    real_targets = torch.ones(clear.size(0), 1)
    real_loss = loss(sigmoid(real_preds), real_targets)
    real_score = torch.mean(real_preds).item()

    fake_clear = generative(cloud)
    fake_preds = discriminator(fake_clear)
    fake_targets = torch.zeros(fake_clear.size(0), 1)
    fake_loss = loss(sigmoid(fake_preds), fake_targets)
    fake_score = torch.mean(fake_preds).item()

    loss = real_loss + fake_loss
    disc_loss = loss.item()
    loss.backward()
    opt_d.step()
    return disc_loss, real_score, fake_score


def train_generator(cloud, opt_g, loss):
    opt_g.zero_grad()

    fake_imgs = generative(cloud)
    disc_preds = discriminator(fake_imgs)
    fake_targets = torch.ones(cloud.size(0), 1)
    loss = loss(disc_preds, fake_targets)
    gen_loss = loss.item()
    loss.backward()
    opt_g.step()
    return gen_loss


def train(epochs, dataloader, opt_d, opt_g, batch_clouds):
    writer = tboard.SummaryWriter('runs')
    loss = nn.BCELoss()
    for epoch in range(epochs):
        gen_loss = 0
        disc_loss = 0
        score_real = 0
        score_fake = 0
        for cloud, clear in dataloader:
            discriminator.train()
            generative.eval()
            loss_d, current_real_score, current_fake_score = train_discriminator(cloud, clear, opt_d, loss)

            discriminator.eval()
            generative.train()
            loss_g = train_generator(cloud, opt_g, loss)

            gen_loss += loss_g
            disc_loss += loss_d

            score_real += current_real_score / len(dataloader)
            score_fake += current_fake_score / len(dataloader)

        generative.eval()
        writer.add_scalars('Loss', {'gen': gen_loss, 'disc': disc_loss}, epoch)
        writer.add_scalars('Scores', {'real': score_real, 'fake': score_fake}, epoch)

        write_imgs(writer, batch_clouds, epoch, generative)

        print("Epoch [{}/{}], loss_g: {:.8f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs, gen_loss, disc_loss, score_real, score_fake))

        if epoch % 50 == 0:
            torch.save(discriminator, f'discriminator-{epoch}.pth')
            torch.save(generative, f'generative-{epoch}.pth')
        else:
            torch.save(discriminator, 'discriminator.pth')
            torch.save(generative, f'generative.pth')


def write_imgs(writer, batch_clouds, epoch, generative):
    fig, axes = plt.subplots(4, 8)
    clouds_axes = axes[:, :4]
    clear_axes = axes[:, 4:]
    for ax, arr_cloud in zip(flatten(clouds_axes), batch_clouds):
        arr_cloud = np.transpose(arr_cloud.detach().numpy(), (1, 2, 0)).astype(np.uint8)
        ax.imshow(arr_cloud)
        ax.axis(False)
    fake_clear = generative(torch.Tensor(batch_clouds)).detach().numpy()
    for ax, fake in zip(flatten(clear_axes), fake_clear):
        fake = (np.transpose(fake, (1, 2, 0)) * 255).astype(np.uint8)
        ax.imshow(fake)
        ax.axis(False)
    plt.tight_layout(pad=0)
    writer.add_figure('Images', fig, epoch)


def main(epochs=10000, lr=0.0002, beta1=0.5):
    clouds = get_clouds()
    batch_clouds = clouds[:16].detach()
    print("CLOUDS LOADED")
    clear = get_clear()
    print("CLEAR LOADED")

    dataloader = get_loader(clear, clouds)

    opt_d = optim.Adam(discriminator.parameters(), lr=lr * 0.1, betas=(beta1, 0.999))
    opt_g = optim.Adam(generative.parameters(), lr=lr, betas=(beta1, 0.999))

    train(epochs, dataloader, opt_d, opt_g, batch_clouds)


def get_loader(clear, clouds):
    dataset = TensorDataset(clouds, clear)
    dataloader = DataLoader(dataset, batch_size=128)
    return dataloader


def get_clear():
    clear = torch.Tensor(np.load('../data/crops/val_clear3.npy'))
    return clear


def get_clouds():
    clouds = torch.Tensor(np.load('../data/crops/val_clouds3.npy'))
    return clouds


main()
