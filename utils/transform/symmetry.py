import torch
from .canonical import view2canonical, canonical2view


def get_symmetrical_points(points: torch.Tensor, dist: torch.Tensor, elev: torch.Tensor, azim: torch.Tensor):
    assert points.ndimension() == 3  # (B, N, 3)
    assert dist.ndimension() == 2  # (B, 1)

    can_points = view2canonical(points, dist, elev, azim)
    sym_points = can_points.clone()
    sym_points[..., 2] = -sym_points[..., 2]

    sym_points = canonical2view(points, dist, elev, azim)
    return sym_points
