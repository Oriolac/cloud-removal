import logging

import mlflow
import torch
from torch import nn
import numpy as np
from skimage.metrics import structural_similarity as ssim


def print_experiment(experiment):
    logging.info("Experiment_id: {}".format(experiment.experiment_id))
    logging.info("Artifact Location: {}".format(experiment.artifact_location))
    logging.info("Tags: {}".format(experiment.tags))
    logging.info("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


def log_model_properties(model, name=""):
    num_model_parameters = sum([p.numel() for p in model.parameters()])
    key = "parameters"
    if name != "":
        key = "{}.{}".format(name, key)
    mlflow.log_param(key, num_model_parameters)
    logging.info("Number of parameters: {}".format(num_model_parameters))


def cuda_info():
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    free = (reserved - allocated)
    logging.info(
        "Total memory {:.3f}, reserved memory: {:.3f} GB, allocated memory: {:.3f} GB, free memory: {:.3f} GB".format(
            total, reserved, allocated, free))


def log_class_attributes_to_mlflow(obj, parent_key=''):
    items = obj.items() if isinstance(obj, dict) else obj.__dict__.items()
    for attr, value in items:
        if attr == "model":
            break
        if not isinstance(value, (int, float, str, list, type(None))) and attr != "kwargs" and attr not in [
            "strategies"]:
            log_class_attributes_to_mlflow(value, f"{parent_key}.{attr}" if parent_key else attr)
        else:
            mlflow.log_params({f"{parent_key}.{attr}" if parent_key else attr: value})


@torch.no_grad()
def calculate_psnr(im1, im2):
    mse = torch.mean((im1 - im2) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def scaled(x):
    min_val, max_val = np.percentile(x, (2, 98))

    # Scale the pixel values to the range of 0-255
    return np.interp(x, (min_val, max_val), (0.01, 255)).astype(np.uint8)


@torch.no_grad()
def calculate_ssim(im1, im2):
    im1 = scaled(np.array(im1.cpu()))
    im2 = scaled(np.array(im2.cpu()))
    return ssim(im1, im2, multichannel=True, data_range=1)


@torch.no_grad()
def calculate_carl(input, target, output, mask):
    input = input[:, 2:]
    no_mask_res = ((torch.ones_like(mask) - mask) * torch.abs((input - output))).mean()
    mask_res = (mask * torch.abs(target - output)).mean()
    return (no_mask_res + mask_res + torch.abs(target - output).mean()).item()

@torch.no_grad()
def calculate_sam(y_true, y_pred):
    mat = y_true * y_pred
    mat = torch.sum(mat, dim=1)
    mat = mat / torch.sqrt(torch.sum(y_true * y_true, dim=1))
    mat = mat / torch.sqrt(torch.sum(y_pred * y_pred, dim=1))

    # Clamp values to avoid values slightly outside the range [-1, 1] due to numerical precision
    mat = mat / torch.sqrt(torch.sum(y_pred * y_pred, dim=1))
    mat = torch.acos(torch.clamp(mat, -1, 1))


    return mat.mean()

def weights_init(neural_net):
    """ Initializes the weights of the neural network
    :param neural_net: (De-)Convolutional Neural Network where weights should be initialized
    """
    for m in neural_net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
