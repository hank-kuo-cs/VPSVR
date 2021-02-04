import torch
import torch.nn as nn


class SphericalCoordinateMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_func = nn.MSELoss()

    def forward(self, pred_elevs: torch.Tensor, pred_azims: torch.Tensor,
                gt_elevs: torch.Tensor,  gt_azims: torch.Tensor,
                w1=0.25, w2=1.0):
        self.check_params(pred_elevs, pred_azims, gt_elevs, gt_azims)

        gt_azims = self.transform_gt_azims(pred_azims, gt_azims)

        return w1 * self.mse_func(pred_elevs, gt_elevs) + w2 * self.mse_func(pred_azims, gt_azims)

    @staticmethod
    def transform_gt_azims(pred_azims: torch.Tensor, gt_azims: torch.Tensor):
        condition1 = torch.abs(pred_azims - gt_azims) > 0.5
        condition2 = pred_azims > gt_azims
        condition3 = pred_azims < gt_azims

        gt_azims = torch.where(condition1 & condition2, gt_azims + 1, gt_azims)
        gt_azims = torch.where(condition1 & condition3, gt_azims - 1, gt_azims)

        return gt_azims

    @staticmethod
    def check_params(pred_elevs: torch.Tensor, pred_azims: torch.Tensor,
                     gt_elevs: torch.Tensor,  gt_azims: torch.Tensor):
        assert pred_elevs.ndimension() == 2  # (B, 1)
        assert pred_azims.ndimension() == 2  # (B, 1)
        assert gt_elevs.ndimension() == 2  # (B, 1)
        assert gt_azims.ndimension() == 2  # (B, 1)
