import torch
from .rotate import rotate_points


def view2canonical(points: torch.Tensor, dists: torch.Tensor, elevs: torch.Tensor, azims: torch.Tensor):
    """
    Transform the points in view-center space to canonical space
    :param points: vertices in view-center space (B x N x 3)
    :param dists: dist of the points (B, 1)
    :param elevs: elev of the points (B, 1)
    :param azims: azim of the points (B, 1)
    :return: canonical_points (torch.Tensor) (B x N x 3)
    """
    assert points.ndimension() == 3  # (B, N, 3)
    assert dists.ndimension() == elevs.ndimension() == azims.ndimension() == 2  # (B, 1)
    elevs, azims = elevs / 360, azims / 360

    B = points.size(0)
    y = torch.tensor([[0, 1, 0]], device=points.device)
    neg_z = torch.tensor([[0, 0, -1]], device=points.device)
    y = torch.repeat_interleave(y, repeats=B, dim=0).float()
    neg_z = torch.repeat_interleave(neg_z, repeats=B, dim=0).float()

    q1 = torch.cat([neg_z, elevs], dim=1)
    y = rotate_points(y.unsqueeze(1), q1).squeeze(1)

    q2 = torch.cat([y, -azims], dim=1)
    points = rotate_points(points, q2)

    q3 = torch.cat([neg_z, -elevs], dim=1)
    points = rotate_points(points, q3)

    dists = torch.repeat_interleave(dists.unsqueeze(2), repeats=points.size(1), dim=1)
    dists = torch.repeat_interleave(dists, repeats=points.size(2), dim=2)

    canonical_points = points * dists

    return canonical_points


def canonical2view(points: torch.Tensor, dists: torch.Tensor, elevs: torch.Tensor, azims: torch.Tensor):
    """
    Transform the points in canonical space to view-center space
    :param points: vertices in canonical space (B x N x 3)
    :param dists: dist of the points (B, 1)
    :param elevs: elev of the points (B, 1)
    :param azims: azim of the points (B, 1)
    :return: view_center_points (torch.Tensor) (B x N x 3)
    """
    assert points.ndimension() == 3  # (B, N, 3)
    assert dists.ndimension() == elevs.ndimension() == azims.ndimension() == 2  # (B, 1)
    elevs, azims = elevs / 360, azims / 360

    B = points.size(0)
    y = torch.repeat_interleave(torch.tensor([[0, 1, 0]], device=points.device), repeats=B, dim=0).float()
    neg_z = torch.repeat_interleave(torch.tensor([[0, 0, -1]], device=points.device), repeats=B, dim=0).float()

    q = torch.cat([neg_z, elevs], dim=1)
    points = rotate_points(points, q)

    y = y.unsqueeze(1)
    y = rotate_points(y, q).squeeze(1)

    q = torch.cat([y, azims], dim=1)
    points = rotate_points(points, q)

    dists = torch.repeat_interleave(dists.unsqueeze(2), repeats=points.size(1), dim=1)
    dists = torch.repeat_interleave(dists, repeats=points.size(2), dim=2)

    points = points / dists

    return points
