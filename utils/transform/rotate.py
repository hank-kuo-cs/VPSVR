import torch

PI = 3.1415927410125732


def rotate_points(points: torch.Tensor, quaternions: torch.Tensor):
    """
    Rotate input points with quaternion.
    :param points: torch.Tensor(B, N, 3)
    :param quaternions: torch.Tensor(B, 4)
        [:, :3] = direction vectors
        [:, 3] = rotate angles between [0, 1] (will be mapped to [0, 2pi])
    :return: rotated points (torch.Tensor(B, N, 3))
    """
    check_points(points)
    check_quaternions(quaternions)

    quaternions = refine_quaternions(quaternions)
    matrices = get_rotation_matrices(quaternions)  # (B, 3, 3)

    points_transpose = points.permute(0, 2, 1)  # (B, 3, N)
    rotated_points = torch.bmm(matrices, points_transpose).permute(0, 2, 1)

    return rotated_points


def get_rotation_matrices(quaternions: torch.Tensor):
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w

    matrices = torch.zeros(quaternions.size(0), 3, 3, dtype=torch.float, requires_grad=True).cuda()

    matrices[:, 0, 0] = x2 - y2 - z2 + w2
    matrices[:, 1, 0] = 2 * (xy + zw)
    matrices[:, 2, 0] = 2 * (xz - yw)
    matrices[:, 0, 1] = 2 * (xy - zw)
    matrices[:, 1, 1] = - x2 + y2 - z2 + w2
    matrices[:, 2, 1] = 2 * (yz + xw)
    matrices[:, 0, 2] = 2 * (xz + yw)
    matrices[:, 1, 2] = 2 * (yz - xw)
    matrices[:, 2, 2] = - x2 - y2 + z2 + w2

    return matrices


def check_points(points: torch.Tensor):
    assert points.ndimension() == 3  # (B, N, 3)
    assert points.size(-1) == 3  # x, y, z


def check_quaternions(quaternions: torch.Tensor):
    assert quaternions.ndimension() == 2  # (B, 4)
    assert quaternions.size(-1) == 4  # direction, angle


def refine_quaternions(quaternions: torch.Tensor):
    q_refine = torch.zeros_like(quaternions)

    directions = quaternions[:, :3]
    angles = torch.div((quaternions[:, 3] % 1) * 2 * PI, 2)
    tmp_angles = torch.sin(angles.unsqueeze(0).view(-1, 1).repeat(1, 3))

    q_refine[:, :3] = torch.mul(directions, tmp_angles)
    q_refine[:, 3] = torch.cos(angles)

    length = torch.norm(q_refine, dim=1)
    q_refine = torch.div(q_refine.T, length).T

    return q_refine
