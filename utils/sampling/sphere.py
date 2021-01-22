import torch


PI = 3.1415927410125732


def sphere_sampling(v: torch.Tensor, num_points: int = 128):
    check_parameters(v, num_points)

    points = reparameterization(num_points, v)
    return points


def check_parameters(v, num_points):
    assert v.ndimension() == 2 and v.size(-1) == 3  # (B, 3)
    assert type(num_points) == int and num_points > 0


def reparameterization(N, v):
    B = v.size(0)

    dist_uniform_samples = torch.ones((B, N, 1)).cuda()
    elev_uniform_samples = -torch.acos(1 - 2 * torch.rand((B, N, 1)).cuda()) + PI * 0.5
    azim_uniform_samples = torch.rand((B, N, 1)).cuda() * 2 * PI

    uniform_points = spherical_to_cartesian(dist_uniform_samples, elev_uniform_samples, azim_uniform_samples)
    repeat_index = torch.tensor([N for i in range(B)]).cuda()
    scaling = torch.repeat_interleave(v, repeat_index, dim=0).view(B, N, 3)

    points = uniform_points * scaling
    return points


def spherical_to_cartesian(dists, elevs, azims):
    xs = dists * torch.cos(elevs) * torch.sin(azims)
    ys = dists * torch.sin(elevs)
    zs = dists * torch.cos(elevs) * torch.cos(azims)

    points = torch.cat([xs, ys, zs], 2)
    return points
