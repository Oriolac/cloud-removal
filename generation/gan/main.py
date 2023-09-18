from collections import defaultdict

import torch
import tqdm
import yaml
from torch.nn import MSELoss

import gan.config
from gan.models import get_model
from data import get_dataloaders
from hyper import get_criterion, get_optimizer, get_metric
from utils import *
import mlflow

torch.set_float32_matmul_precision('medium')


class Trainer:

    def __init__(self, generator, discriminator, device, criterion_generator, criterion_discriminator,
                 optimizer_generator, optimizer_discriminator, checkpoint: gan.config.Checkpoint,
                 metrics):
        self.generator: nn.Module = generator
        self.discriminator: nn.Module = discriminator
        self.device = device
        self.criterion_generator = criterion_generator
        self.criterion_discriminator = criterion_discriminator
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.checkpoint = checkpoint
        self.metrics = metrics

    def training_loop(self, epoch, train_loader, val_loader, test_loader):
        generator_loss = 0
        discriminator_loss = 0
        accuracy_discriminator = 0
        for i, (cloudy, cloudless) in tqdm.tqdm(enumerate(train_loader)):
            cloudy = cloudy.to(self.device)
            cloudless = cloudless.to(self.device)

            t = i + epoch * len(train_loader.dataset)
            batch_discriminator_loss, accuracy_batch = self.train_discriminator(cloudy, cloudless, t)
            accuracy_discriminator += accuracy_batch
            batch_generator_loss, mse_loss = self.train_generator(cloudy, cloudless, t)

            generator_loss += batch_generator_loss + mse_loss
            discriminator_loss += batch_discriminator_loss

            if (i + 5000) % 10000 == 0:
                self.check_loaders(epoch, t, val_loader, test_loader)
                mlflow.log_metric("accuracy_discriminator", accuracy_discriminator / i, t)

        return generator_loss, discriminator_loss

    def train_generator(self, cloudy, cloudless, t):
        self.discriminator.eval()
        self.generator.train()

        self.generator.zero_grad()

        label = torch.ones(cloudless.size(0), device=self.device)
        fake_cloudless = self.generator(cloudy)
        output = self.discriminator(fake_cloudless).view(-1)
        criterion_loss_generator = self.criterion_generator(output, label)
        mse_loss_generator = MSELoss()(fake_cloudless, cloudless)
        (criterion_loss_generator + mse_loss_generator).backward()
        self.optimizer_generator.step()

        mlflow.log_metric("batch_loss_generator", criterion_loss_generator.item(), t)
        mlflow.log_metric("mse_loss_generator", mse_loss_generator.item(), t)
        mlflow.log_metric("batch_output_discriminator_generations", output.detach().mean().item(), t)
        return criterion_loss_generator.item(), mse_loss_generator.item()

    def train_discriminator(self, cloudy, cloudless, t):
        self.generator.eval()
        self.discriminator.train()

        self.discriminator.zero_grad()

        loss_real, tp, output_real = self.train_discriminator_real(cloudless)
        loss_fake, tn, output_fake = self.train_discriminator_fake(cloudy, cloudless)

        loss_discriminator = loss_real + loss_fake
        self.optimizer_discriminator.step()

        mlflow.log_metric("batch_loss_real", loss_real.item(), t)
        mlflow.log_metric("batch_loss_fake", loss_fake.item(), t)
        mlflow.log_metric("batch_fake", output_fake, t)
        mlflow.log_metric("batch_real", output_real, t)
        mlflow.log_metric("batch_loss_discriminator", loss_discriminator.item(), t)
        accuracy_discriminator = (tp + tn) / (cloudy.size(0) * 2)
        mlflow.log_metric("batch_accuracy_discriminator", accuracy_discriminator, t)

        return loss_discriminator.item(), accuracy_discriminator

    def train_discriminator_fake(self, cloudy, cloudless):
        fake_label = torch.zeros(cloudless.size(0), device=self.device)
        fake_cloudless = self.generator(cloudy)
        output = self.discriminator(fake_cloudless).view(-1)
        loss_fake = self.criterion_discriminator(output, fake_label)
        loss_fake.backward()
        tn = (output.detach() < 0.5).sum().item()
        return loss_fake, tn, output.detach().mean().item()

    def train_discriminator_real(self, cloudless):
        label = torch.ones(cloudless.size(0), device=self.device)
        output = self.discriminator(cloudless).view(-1)
        loss_real = self.criterion_discriminator(output, label)
        loss_real.backward()
        tp = (output.detach() > 0.5).sum().item()
        return loss_real, tp, output.detach().mean().item()

    def eval_loop(self, dataloader):
        metrics_ = defaultdict(lambda: 0.0)
        for cloudy, cloudless in tqdm.tqdm(dataloader):
            self.generator.eval()
            cloudy = cloudy.to(self.device)
            cloudless = cloudless.to(self.device)
            fake_cloudless = self.generator(cloudy)
            for metric_name, metric_func in self.metrics.items():
                metrics_[metric_name] += metric_func(cloudless, fake_cloudless).item()
        for metric_name in metrics_.keys():
            metrics_[metric_name] = metrics_[metric_name] / len(dataloader.dataset)
        return metrics_

    def start(self, train_loader, val_loader, test_loader, max_epochs):
        self.max_ssim = float("-inf")
        self.min_val_loss = float("inf")
        for epoch in range(max_epochs):
            logging.info("[{}]\t Starting epoch [{}]".format(epoch, epoch))
            generator_loss, discriminator_loss = self.training_loop(epoch, train_loader, val_loader, test_loader)
            cuda_info()

            self.check_loaders(epoch, epoch * len(train_loader.dataset), val_loader, test_loader)
            logging.info(
                '[{}]\t Generator Loss: {:.10f}\t Discriminator loss {:.10f}'.format(epoch, generator_loss,
                                                                                     discriminator_loss))
            mlflow.log_metric("generator_loss", generator_loss, epoch)
            mlflow.log_metric("discriminator_loss", discriminator_loss, epoch)

    def check_loaders(self, epoch, t, test_loader, val_loader):
        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            val_metrics = self.eval_loop(val_loader)
            test_metrics = self.eval_loop(test_loader)
        cuda_info()
        if self.min_val_loss > val_metrics["MSELoss"]:
            logging.info(
                "[{} - {}]\t Saving chk since loss {:.10g} < {:.10g}".format(epoch, t, val_metrics["MSELoss"],
                                                                             self.min_val_loss))
            mlflow.pytorch.log_model(self.generator, self.checkpoint.path + self.checkpoint.model,
                                     metadata={"epoch": epoch})
            self.min_val_loss = val_metrics["MSELoss"]
        if self.max_ssim < val_metrics["SSIM"]:
            logging.info(
                "[{} - {}]\t Saving chk since ssim {:.10g} > {:.10g}".format(epoch, t, val_metrics["SSIM"],
                                                                             self.max_ssim))
            mlflow.pytorch.log_model(self.generator, self.checkpoint.path + self.checkpoint.model,
                                     metadata={"epoch": epoch, "i": t})
            self.max_ssim = val_metrics["SSIM"]
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value, t)
            logging.info('[{} - {}]\t Val {} {:.10f}'.format(epoch, t, metric_name, metric_value))
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value, t)
            logging.info('[{} - {}]\t Test {} {:.10f}'.format(epoch, t, metric_name, metric_value))


def main(config: str):
    mlflow.autolog()
    mlflow.set_tracking_uri("/runs/")
    experiment = mlflow.set_experiment("Cloud Removal CNN")

    print_experiment(experiment)

    torch.cuda.empty_cache()
    with open(config, 'r') as file:
        yaml_content = yaml.safe_load(file)
    config = gan.config.populate_classes(yaml_content)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0") if config.gpu == "all" else torch.device(f"cuda:{config.gpu}")
    cuda_info()
    train_loader, val_loader, test_loader = get_dataloaders(config.data)
    cuda_info()
    generator = get_model(config.train.generator, device)
    discriminator = get_model(config.train.discriminator, device)
    if config.gpu == "all":
        generator = torch.nn.DataParallel(generator,
                                          device_ids=[dev for dev in range(torch.cuda.device_count())])
    cuda_info()
    criterion_generator = get_criterion(config.train.criterion_generator)
    optimizer_generator = get_optimizer(generator, config.train.optimizer_generator)
    criterion_discriminator = get_criterion(config.train.criterion_discriminator)
    optimizer_discriminator = get_optimizer(discriminator, config.train.optimizer_discriminator)
    metrics = {name: get_metric(name, kwargs) for name, kwargs in config.train.metrics.items()}
    trainer = Trainer(generator, discriminator, device, criterion_generator, criterion_discriminator,
                      optimizer_generator, optimizer_discriminator, config.train.checkpoint, metrics)
    with mlflow.start_run():
        log_class_attributes_to_mlflow(config)
        log_model_properties(generator, name="generator")
        log_model_properties(discriminator, name="discriminator")
        trainer.start(train_loader, val_loader, test_loader, config.train.epochs)
