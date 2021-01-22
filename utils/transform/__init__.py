import torch
from .rotate import rotate_points
from .translate import translate_points


def transform_points(points: torch.Tensor, q: torch.Tensor, t: torch.Tensor):
    check_parameters(points, q, t)

    return translate_points(rotate_points(points, q), t)


def check_parameters(points: torch.Tensor, q: torch.Tensor, t: torch.Tensor):
    assert points.ndimension() == 3  # (B, N, 3)
    assert points.size(-1) == 3
    B = points.size(0)

    assert q.size() == (B, 4)
    assert t.size() == (B, 3)
