import torch
from .image import concat_images, denormlize_image


def save_depth_result(rgb: torch.Tensor, predict_depth: torch.Tensor, gt_depth: torch.Tensor, save_path: str):
    assert rgb.ndimension() == 3  # (3, H, W)
    assert predict_depth.ndimension() == 3  # (1, H, W)
    assert gt_depth.ndimension() == 3  # (1, H, W)

    rgb = denormlize_image(rgb)

    concated_img = concat_images([rgb, predict_depth, gt_depth])
    concated_img.save(save_path)
