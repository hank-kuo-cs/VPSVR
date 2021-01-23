import torch
import torch.nn as nn


def get_local_features(vertices: torch.Tensor, imgs: torch.Tensor, perceptual_features: list):
    bounds = get_bound_of_images(imgs)
    local_features = perceptual_feature_pooling(perceptual_features, vertices, bounds)

    return local_features


def get_bound_of_images(imgs: torch.Tensor):
    assert imgs.ndimension() == 4  # (B, C, H, W)
    h, w = imgs.size(2), imgs.size(3)
    bounds = torch.zeros((imgs.size(0), 4)).cuda()
    bounds[:, 1] = w
    bounds[:, 3] = h

    for b in range(imgs.size(0)):
        img = imgs[b]
        mask = img.sum(0)

        x_any = (mask > 0.03).any(0)  # set  0.03 to prevent mask has some low noise value to affect the result
        y_any = (mask > 0.03).any(1)

        for i in range(w):
            j = w - i - 1

            if x_any[i] and bounds[b, 0] == 0:
                bounds[b, 0] = i

            if x_any[j] and bounds[b, 1] == w:
                bounds[b, 1] = j

            if bounds[b, 0] > 0 and bounds[b, 1] < w:
                break

        for i in range(h):
            j = h - i - 1

            if y_any[i] and bounds[b, 2] == 0:
                bounds[b, 2] = i

            if y_any[j] and bounds[b, 3] == h:
                bounds[b, 3] = j

            if bounds[b, 2] > 0 and bounds[b, 3] < h:
                break

    # normalize bounds value to [-1, 1]
    bounds[:, :2] = bounds[:, :2] / w * 2 - 1
    bounds[:, 2: 4] = bounds[:, 2: 4] / h * 2 - 1

    return bounds  # (B, 4)


def perceptual_feature_pooling(perceptual_features: list, points: torch.Tensor, bounds: torch.Tensor):
    assert points.ndimension() == 3.  # (B, N, 3)
    assert bounds.ndimension() == 2.  # (B, 4)

    bounds = bounds[:, None, :].repeat(1, points.size(1), 1)  # (B, N, 4)

    grids = torch.zeros((points.size(0), points.size(1), 2), dtype=torch.float, device=points.device)
    max_zs = torch.zeros((points.size(0), points.size(1), 1), dtype=torch.float, device=points.device)
    min_zs, max_ys, min_ys = max_zs.clone(), max_zs.clone(), max_zs.clone()

    for b in range(points.size(0)):
        instance_points = points[b]
        max_pos, min_pos = instance_points.max(0)[0], instance_points.min(0)[0]
        max_zs[b, :, 0], min_zs[b, :, 0] = max_pos[2], min_pos[2]
        max_ys[b, :, 0], min_ys[b, :, 0] = max_pos[1], min_pos[1]

    grids[..., 0] = bounds[..., 0] + (1 - (points[..., 2] - min_zs[..., 0]) / (max_zs[..., 0] - min_zs[..., 0])) * (
                bounds[..., 1] - bounds[..., 0])
    grids[..., 1] = bounds[..., 2] + (1 - (points[..., 1] - min_ys[..., 0]) / (max_ys[..., 0] - min_ys[..., 0])) * (
                bounds[..., 3] - bounds[..., 2])

    grids = grids[:, None, :, :]

    pooling_features = []

    for layer_features in perceptual_features:
        layer_pooling_features = nn.functional.grid_sample(layer_features, grids, align_corners=True)
        pooling_features.append(layer_pooling_features)

    pooling_features = torch.cat(pooling_features, 1).view(points.size(0), -1, points.size(1)).permute(0, 2, 1)
    return pooling_features  # (B, N, C)
