import logging

import mlflow
import numpy as np
import torch
import tqdm
import yaml
from diffusers import DDPMScheduler

from data import get_dataloaders
from hyper import get_criterion, get_optimizer
from utils import print_experiment, cuda_info, log_class_attributes_to_mlflow, log_model_properties, calculate_psnr, \
    calculate_ssim
from dif.models import MODEL
import dif.config

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


class Trainer:

    def __init__(self, model, device, criterion, optimizer, checkpoint: dif.config.Checkpoint):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint = checkpoint

    def start(self, train_loader, val_loader, test_loader, max_epochs):
        max_ssim = float("-inf")
        min_val_loss = float("inf")
        for epoch in range(max_epochs):
            logging.info("[{}]\t Starting epoch [{}]".format(epoch, epoch))
            train_loss = self.training_loop(train_loader)
            cuda_info()
            with torch.no_grad():
                val_loss, val_ssim, val_psnr = self.eval_loop(val_loader)
                test_loss, test_ssim, test_psnr = self.eval_loop(test_loader)
            cuda_info()
            mlflow.log_metric("norm_train_loss", train_loss, epoch)
            mlflow.log_metric("norm_val_loss", val_loss, epoch)
            mlflow.log_metric("norm_test_loss", test_loss, epoch)
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

    def training_loop(self, dataloader):
        self.model.train()
        loss = 0
        for image, target in tqdm.tqdm(dataloader):
            self.model.zero_grad()
            image = image.to(self.device)
            target = target.to(self.device)
            noise = torch.randn(image.shape).to(self.device)
            timesteps = torch.randint(
                0, 1000, (image.size(0),), device=self.device
            ).long()
            noisy_images = noise_scheduler.add_noise(image, noise, timesteps)
            noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
            batch_loss = self.criterion(noise_pred, target)
            batch_loss.backward()
            batch_loss_item = batch_loss.item() * image.size(0) / len(dataloader.dataset.df)

            loss += batch_loss_item
        return loss

    def eval_loop(self, dataloader):
        self.model.eval()
        loss = 0
        ssims = []
        psnrs = []
        for image, target in tqdm.tqdm(dataloader):
            image = image.to(self.device)
            target = target.to(self.device)
            ssims_batch, psnsrs_batch = [], []
            noise = torch.randn(image.shape).to(self.device)
            timesteps = torch.randint(
                0, 1000, (image.size(0),), device=self.device
            ).long()
            noisy_images = noise_scheduler.add_noise(image, noise, timesteps)
            y_hat = self.model(noisy_images, timesteps, return_dict=False)[0]
            batch_loss = self.criterion(y_hat, target)
            loss += batch_loss.item() * image.size(0) / len(dataloader.dataset.df)
            for i in range(image.size(0)):
                ssims_batch.append(calculate_ssim(target[i], y_hat[i]))
                psnsrs_batch.append(calculate_psnr(target[i], y_hat[i]))
            ssims.append(float(np.mean(ssims_batch)))
            psnrs.append(float(np.mean(psnsrs_batch)))
        return loss, float(np.mean(ssims)), float(np.mean(psnrs))


def main(config: str):
    mlflow.autolog()
    mlflow.set_tracking_uri("/runs/")
    experiment = mlflow.set_experiment("Cloud Removal CNN")

    print_experiment(experiment)

    torch.cuda.empty_cache()
    with open(config, 'r') as file:
        yaml_content = yaml.safe_load(file)
    model = MODEL
    config = dif.config.populate_classes(yaml_content, model)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0") if config.gpu == "all" else torch.device(f"cuda:{config.gpu}")
    cuda_info()
    train_loader, val_loader, test_loader = get_dataloaders(config.data)
    cuda_info()
    model = model.to(device)
    criterion = get_criterion(config.train.criterion)
    optimizer = get_optimizer(model, config.train.optimizer)
    trainer = Trainer(model, device, criterion, optimizer, config.train.checkpoint)
    with mlflow.start_run():
        log_class_attributes_to_mlflow(config)
        log_model_properties(model)
        trainer.start(train_loader, val_loader, test_loader, config.train.epochs)
