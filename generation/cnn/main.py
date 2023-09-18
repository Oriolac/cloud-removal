import sys

import tqdm
import yaml
from torch.utils.data import DataLoader

import cnn.config
from cnn import get_model
from data import get_dataloaders
from hyper import get_criterion, get_optimizer
from utils import *

torch.set_float32_matmul_precision('medium')


class Trainer:

    def __init__(self, model, device, criterion, optimizer, checkpoint: cnn.config.Checkpoint):
        self.model: nn.Module = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint = checkpoint

    def common_step(self, x, y):
        y, y_hat = self.inference(x, y)
        if torch.isnan(y_hat).any():
            logging.error("Warning: Found nan or inf in model output")
        loss = self.criterion(y_hat, y)
        return loss

    def inference(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self.model(x)
        return y, y_hat

    def training_loop(self, dataloader: DataLoader):
        self.model.train()
        loss = 0
        for i, (image, target) in tqdm.tqdm(enumerate(dataloader)):
            self.model.zero_grad()
            batch_loss = self.common_step(image, target)
            batch_loss.backward()
            self.optimizer.step()
            batch_loss_item = batch_loss.item() * image.size(0) / len(dataloader.dataset.df)

            loss += batch_loss_item
        return loss

    def eval_loop(self, dataloader: DataLoader):
        self.model.eval()
        loss = 0
        ssims = []
        psnrs = []
        for img, target in tqdm.tqdm(dataloader):
            ssims_batch, psnsrs_batch = [], []
            target, y_hat = self.inference(img, target)
            batch_loss = self.criterion(y_hat, target)
            loss += batch_loss.item() * img.size(0) / len(dataloader.dataset.df)
            for i in range(img.size(0)):
                ssims_batch.append(calculate_ssim(target[i], y_hat[i]))
                psnsrs_batch.append(calculate_psnr(target[i], y_hat[i]))
            ssims.append(float(np.mean(ssims_batch)))
            psnrs.append(float(np.mean(psnsrs_batch)))
        return loss, float(np.mean(ssims)), float(np.mean(psnrs))

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


def main(config: str):
    mlflow.autolog()
    mlflow.set_tracking_uri("/runs/")
    experiment = mlflow.set_experiment("Cloud Removal CNN")

    print_experiment(experiment)

    torch.cuda.empty_cache()
    with open(config, 'r') as file:
        yaml_content = yaml.safe_load(file)
    config = cnn.config.populate_classes(yaml_content)
    torch.manual_seed(config.seed)
    if config.gpu != "cpu":
        device = torch.device("cuda:0") if config.gpu == "all" else torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")
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
