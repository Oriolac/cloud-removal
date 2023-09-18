
import torch

def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return torch.mean(torch.abs(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))

def cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return torch.mean((y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]) ** 2)

def cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return torch.sqrt(torch.mean((y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]) ** 2))

def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    return torch.mean(torch.sqrt(torch.mean((y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]) ** 2, dim=[2, 3])))

def cloud_PSNR(y_true, y_pred):
    """Computes the PSNR over the full image."""
    mse = torch.mean((y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]) ** 2)
    return 10 * torch.log10(1 / mse)

def get_sam(y_true, y_predict):
    """Computes the SAM array."""
    mat = y_true * y_predict
    mat = torch.sum(mat, dim=1)
    mat = mat / torch.sqrt(torch.sum(y_true * y_true, dim=1))
    mat = mat / torch.sqrt(torch.sum(y_predict * y_predict, dim=1))
    mat = torch.acos(torch.clamp(mat, -1, 1))
    
    return mat.mean()


def cloud_mean_sam(y_true, y_predict):
    """Computes the SAM over the full image."""
    mat = get_sam(y_true[:, 0:13, :, :], y_predict[:, 0:13, :, :])

    return torch.mean(mat)


def cloud_mean_sam_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    target = y_true[:, 0:13, :, :]
    predicted = y_pred[:, 0:13, :, :]

    if torch.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    sam = get_sam(target, predicted)
    sam = sam.expand(1)
    sam = torch.sum(cloud_cloudshadow_mask * sam) / torch.sum(cloud_cloudshadow_mask)

    return sam

def cloud_mean_sam_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = torch.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]

    if torch.sum(clearmask) == 0:
        return 0.0

    sam = get_sam(input_cloudy, predicted)
    sam = sam.expand(1)
    sam = torch.sum(clearmask * sam) / torch.sum(clearmask)

    return sam


def cloud_mean_absolute_error_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = torch.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]

    if torch.sum(clearmask) == 0:
        return 0.0

    clti = clearmask * torch.abs(predicted - input_cloudy)
    clti = torch.sum(clti) / (torch.sum(clearmask) * 13)

    return clti


def cloud_mean_absolute_error_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    target = y_true[:, 0:13, :, :]

    if torch.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    ccmaec = cloud_cloudshadow_mask * K.abs(predicted - target)
    ccmaec = K.sum(ccmaec) / (K.sum(cloud_cloudshadow_mask) * 13)

    return ccmaec