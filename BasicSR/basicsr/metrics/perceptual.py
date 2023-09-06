import torch
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_percep_loss(img_tensor, img2_tensor, percep_loss_fn, **kwargs):
    with torch.no_grad():
        result = percep_loss_fn(img_tensor, img2_tensor, normalize=True).squeeze()
        return result