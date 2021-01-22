import torch


def cuboid_sampling(v: torch.Tensor, num_points: int = 128):
    check_parameters(v, num_points=num_points)
    w, h, d = v[:, 0], v[:, 1], v[:, 2]

    if w.ndimension() == 1:
        w, h, d = w.unsqueeze(1), h.unsqueeze(1), d.unsqueeze(1)

    B = w.size(0)

    num_points_each_face = get_faces_points(w, h, d, num_points)  # (B, 6)

    points = reparameterization(B, num_points, w, h, d)
    points = map_points_onto_surfaces(points, B, w, h, d, num_points_each_face)

    return points


def check_parameters(v, num_points):
    assert v.ndimension() == 2 and v.size(-1) == 3  # (B, 3)
    assert type(num_points) == int and num_points > 0


def get_faces_points(w, h, d, num_points):
    """
    Calculate the number of points each face has based on their area.
    :param w: width (torch.Tensor(B, 1))
    :param h: height (torch.Tensor(B, 1))
    :param d: depth (torch.Tensor(B, 1))
    :param num_points: number of points (int)
    :return:
    """
    B = w.size(0)
    hd, dw, wh = h * d, d * w, w * h

    faces_area = torch.cat([hd, hd, dw, dw, wh, wh], 1)  # (B, 6)
    total_area = (hd + dw + wh) * 2  # (B, 1)

    faces_weight = faces_area / total_area
    faces_points = (torch.full_like(faces_weight, num_points) * faces_weight).round().int()

    last_face_points = torch.tensor(num_points, dtype=torch.int).repeat(B).cuda()
    for i in range(5):
        last_face_points -= faces_points[:, i]
    faces_points[:, -1] = last_face_points

    return faces_points


def reparameterization(B, N, w, h, d):
    """
    Sample points in the whole volume of cuboid.
    :param B: batch size (int)
    :param N: number of points (int)
    :param w: width (torch.Tensor(B, 1))
    :param h: height (torch.Tensor(B, 1))
    :param d: depth (torch.Tensor(B, 1))
    :return: sampled points in the cuboid (torch.Tensor(B, N, 3))
    """
    uniform_samples = -1 + 2 * torch.rand((B, N, 3), dtype=torch.float).cuda()
    cuboid = torch.cat([w, h, d], 1)
    repeat_index = torch.tensor([N for i in range(B)]).cuda()
    scaling = torch.repeat_interleave(cuboid, repeat_index, dim=0).view(B, N, 3)

    return uniform_samples * scaling


def map_points_onto_surfaces(points, B, w, h, d, num_points_each_face):
    """
    Map sampled points on to surfaces of cuboid
    :param points: Sampled points in the whole cuboid (torch.Tensor(B, N, 3))
    :param B: batch size (int)
    :param w: width (torch.Tensor(B, 1))
    :param h: height (torch.Tensor(B, 1))
    :param d: depth (torch.Tensor(B, 1))
    :param num_points_each_face: How many number of points each face has (torch.Tensor(B, 6))
    :return: sampled points on surfaces (torch.Tensor(B, N, 3))
    """
    # faces: [w, ±h, ±d], [-w, ±h, ±d], [±w, h, ±d], [±w, -h, ±d], [±w, ±h, +d], [±w, ±h, -d]
    now_indices = torch.zeros(B, dtype=torch.int).cuda()

    for i in range(6):
        num_points = num_points_each_face[:, i]  # (B, 1)

        fix_index = i // 2
        sign = -1 if i % 2 else 1
        value = sign * (w if fix_index == 0 else (h if fix_index == 1 else d))

        for b in range(B):
            now, num = now_indices[b].item(), num_points[b].item()
            points[b, now: now + num, fix_index] = value[b].expand([1, num])

        now_indices += num_points

    return points