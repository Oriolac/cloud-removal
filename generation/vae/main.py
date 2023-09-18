import logging

import mlflow
import numpy as np
import torch
import tqdm
import yaml
from torch import nn
from torch.utils.data import DataLoader

import vae
import vae.config
from data import get_dataloaders
from hyper import get_criterion, get_optimizer
from utils import print_experiment, cuda_info, log_class_attributes_to_mlflow, log_model_properties, calculate_ssim, \
    calculate_psnr
from vae.models import get_model

torch.set_float32_matmul_precision('medium')


class Trainer:

    def __init__(self, model, device, criterion, optimizer, checkpoint: vae.config.Checkpoint):
        self.model: nn.Module = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint = checkpoint

    def common_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat, mu, log_var = self.inference(x, y)
        loss = self.criterion(y_hat, y)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return loss, kl_divergence

    def inference(self, x, y):
        y_hat, mu, log_var = self.model(x)
        return y_hat, mu, log_var

    def training_loop(self, dataloader: DataLoader):
        self.model.train()
        recon_loss = 0
        kl_loss = 0
        for image, target in tqdm.tqdm(dataloader):
            recon_loss_batch, kl_loss_batch = self.common_step(image, target)
            batch_loss = recon_loss_batch + kl_loss_batch
            batch_loss.backward()
            self.optimizer.step()

            recon_loss += recon_loss_batch.item() * image.size(0) / len(dataloader.dataset.df)
            kl_loss += kl_loss_batch.item() * image.size(0) / len(dataloader.dataset.df)
        return recon_loss, kl_loss

    def eval_loop(self, dataloader: DataLoader):
        self.model.eval()
        recon_loss = 0
        kl_loss = 0
        ssims = []
        psnrs = []
        for img, target in tqdm.tqdm(dataloader):
            ssims_batch, psnsrs_batch = [], []
            x = img.to(self.device)
            y = target.to(self.device)
            y_hat, mu, log_var = self.inference(x, y)
            recon_loss_batch = self.criterion(y_hat, y)
            kl_loss_batch = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            recon_loss += recon_loss_batch.item() * x.size(0) / len(dataloader.dataset.df)
            kl_loss += kl_loss_batch.item() * x.size(0) / len(dataloader.dataset.df)
            for i in range(img.size(0)):
                ssims_batch.append(calculate_ssim(y[i], y_hat[i]))
                psnsrs_batch.append(calculate_psnr(y[i], y_hat[i]))
            ssims.append(float(np.mean(ssims_batch)))
            psnrs.append(float(np.mean(psnsrs_batch)))
        return recon_loss, kl_loss, float(np.mean(ssims)), float(np.mean(psnrs))

    def start(self, train_loader, val_loader, test_loader, max_epochs):
        max_ssim = float("-inf")
        min_val_loss = float("inf")
        for epoch in range(max_epochs):
            logging.info("[{}]\t Starting epoch [{}]".format(epoch, epoch))
            mse_train_loss, kl_train_loss = self.training_loop(train_loader)
            train_loss = mse_train_loss + kl_train_loss
            cuda_info()
            with torch.no_grad():
                mse_val_loss, kl_val_loss, val_ssim, val_psnr = self.eval_loop(val_loader)
                val_loss = mse_val_loss + kl_val_loss
                mse_test_loss, kl_test_loss, test_ssim, test_psnr = self.eval_loop(test_loader)
                test_loss = mse_test_loss + kl_test_loss
            cuda_info()
            mlflow.log_metric("mse_train_loss", mse_train_loss, epoch)
            mlflow.log_metric("mse_val_loss", mse_val_loss, epoch)
            mlflow.log_metric("mse_test_loss", mse_test_loss, epoch)
            mlflow.log_metric("kl_train_loss", kl_train_loss, epoch)
            mlflow.log_metric("kl_val_loss", kl_val_loss, epoch)
            mlflow.log_metric("kl_test_loss", kl_test_loss, epoch)

            mlflow.log_metric("val_ssim", val_ssim, epoch)
            mlflow.log_metric("test_ssim", test_ssim, epoch)
            mlflow.log_metric("val_psnr", val_psnr, epoch)
            mlflow.log_metric("test_psnr", test_psnr, epoch)
            logging.info(
                '[{}]\t Train Loss: {:.10f}\t Val loss {:.10f}\t Test loss {:.10f}'.format(epoch,
                                                                                           train_loss,
                                                                                           val_loss, test_loss))

            logging.info('[{}]\t Val SSIM {:.10f}\t Test SSIM {:.10f}'.format(epoch, val_ssim, test_ssim))
            logging.info('[{}]\t Val PSNR {:.10f}\t Test PSNR {:.10f}'.format(epoch, val_psnr, test_psnr))

            if min_val_loss > val_loss:
                logging.info("[{}]\t Saving chk since loss {:.10g} < {:.10g}".format(epoch, val_loss, min_val_loss))
                mlflow.pytorch.log_model(self.model, self.checkpoint.path + self.checkpoint.model,
                                         metadata={"epoch": epoch})
                min_val_loss = val_loss
            if max_ssim < val_ssim:
                logging.info("[{}]\t Saving chk since ssim {:.10g} > {:.10g}".format(epoch, val_ssim, max_ssim))
                mlflow.pytorch.log_model(self.model, self.checkpoint.path + self.checkpoint.model,
                                         metadata={"epoch": epoch})
                max_ssim = val_ssim


def main(config: str):
    mlflow.autolog()
    mlflow.set_tracking_uri("/runs/")
    experiment = mlflow.set_experiment("Cloud Removal CNN")

    print_experiment(experiment)

    torch.cuda.empty_cache()
    with open(config, 'r') as file:
        yaml_content = yaml.safe_load(file)
    config = vae.config.populate_classes(yaml_content)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0") if config.gpu == "all" else torch.device(f"cuda:{config.gpu}")
    cuda_info()
    train_loader, val_loader, test_loader = get_dataloaders(config.data)
    cuda_info()
    model = get_model(config.train, device)
    if config.gpu == "all":
        model = torch.nn.DataParallel(model,
                                      device_ids=[dev for dev in range(torch.cuda.device_count())])
    cuda_info()
    criterion = get_criterion(config.train.criterion)
    optimizer = get_optimizer(model, config.train.optimizer)
    trainer = Trainer(model, device, criterion, optimizer, config.train.checkpoint)
    with mlflow.start_run():
        log_class_attributes_to_mlflow(config)
        log_model_properties(model)
        trainer.start(train_loader, val_loader, test_loader, config.train.epochs)
