from functools import reduce

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from model import autoencoderCNN
import torch.utils.tensorboard as tboard
from torch import nn

flatten = lambda z: reduce(lambda x, y: list(x) + list(y), z)


def write_imgs(writer, batch_clouds, epoch, model):
    fig, axes = plt.subplots(4, 8)
    clouds_axes = axes[:, :4]
    clear_axes = axes[:, 4:]
    for ax, arr_cloud in zip(flatten(clouds_axes), batch_clouds):
        arr_cloud = np.transpose(arr_cloud.detach().numpy(), (1, 2, 0)).astype(np.uint8)
        ax.imshow(arr_cloud)
        ax.axis(False)
    fake_clear = model(torch.Tensor(batch_clouds)).detach().numpy()
    for ax, fake in zip(flatten(clear_axes), fake_clear):
        fake = (np.transpose(fake, (1, 2, 0)) * 255).astype(np.uint8)
        ax.imshow(fake)
        ax.axis(False)
    plt.tight_layout(pad=0)
    writer.add_figure('Images', fig, epoch)


def train(epochs, trainloader, testloader, opt, batch_clouds):
    writer = tboard.SummaryWriter('autoencoder')
    criterion = nn.BCELoss()
    test_loss_min = np.inf
    for epoch in range(epochs):
        train_loss, test_loss = 0, 0

        autoencoderCNN.train()
        for cloud, clear in trainloader:
            opt.zero_grad()
            fake = autoencoderCNN(cloud)
            loss = criterion(fake, clear)
            train_loss += loss.item() * cloud.size(0)
            criterion.backward()
            opt.step()
        autoencoderCNN.eval()

        for cloud, clear in testloader:
            output = autoencoderCNN(cloud)
            loss = criterion(output, clear)
            test_loss += loss.item() * cloud.size(0)

        train_loss = train_loss / len(trainloader.sampler)
        test_loss = test_loss / len(testloader.sampler)

        writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, epoch)

        write_imgs(writer, batch_clouds, epoch, autoencoderCNN)

        print("Epoch {} \t Train Loss: {:.6f} \t Test Loss: {:.6f}".format(epoch, train_loss, test_loss))

        if test_loss <= test_loss_min:
            print("Saving...")
            torch.save(autoencoderCNN, 'autoencoder.pth')
            test_loss_min = test_loss


def main(epochs=100, lr=0.0001, beta1=0.5):
    clouds = get_clouds()
    batch_clouds = clouds[:16].detach()
    print("CLOUDS LOADED")
    clear = get_clear()
    print("CLEAR LOADED")

    size_data = clear.size(0)
    train_length = size_data // 5
    trainloader = get_loader(clear[:train_length], clouds[:train_length])
    testloader = get_loader(clear[train_length:], clouds[trainloader:])

    opt = optim.Adam(autoencoderCNN.parameters(), lr=lr, betas=(beta1, 0.999))

    train(epochs, trainloader, testloader, opt, batch_clouds)


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
