import torch
import torch.nn as nn


class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, each_batch=False, w1=1.0, w2=1.0) -> torch.Tensor:
        self.check_parameters(points1)
        self.check_parameters(points2)

        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)

        dist_min1, _ = torch.min(dist1, dim=2)
        dist_min2, _ = torch.min(dist2, dim=2)

        loss1 = dist_min1.mean(1)
        loss2 = dist_min2.mean(1)

        loss = w1 * loss1 + w2 * loss2

        return loss if each_batch else loss.mean()

    @staticmethod
    def check_parameters(points: torch.Tensor):
        assert points.ndimension() == 3  # (B, N, 3)
        assert points.size(-1) == 3
