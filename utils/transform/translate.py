import torch


def translate_points(points: torch.Tensor, translations: torch.Tensor):
    check_points(points)
    check_translations(translations)

    return points + translations.unsqueeze(1).expand_as(points)


def check_points(points: torch.Tensor):
    assert points.ndimension() == 3  # (B, N, 3)
    assert points.size(-1) == 3  # x, y, z


def check_translations(translations: torch.Tensor):
    assert translations.ndimension() == 2  # (B, 3)
    assert translations.size(-1) == 3  # x, y, z
