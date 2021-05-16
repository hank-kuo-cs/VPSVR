import torch
import torch.nn as nn


def get_sobel(depth: torch.Tensor):
    assert depth.ndimension() == 4  # (B, 1, H, W)
    n = depth.size(1)
    x_kernel = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).expand(1, n, 3, 3).cuda()
    y_kernel = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).expand(1, n, 3, 3).cuda()

    conv = nn.Conv2d(n, 1, kernel_size=3, stride=1, padding=1, bias=False)

    conv.weight = torch.nn.Parameter(x_kernel, requires_grad=False)
    sobel_x = conv(depth)

    conv.weight = torch.nn.Parameter(y_kernel, requires_grad=False)
    sobel_y = conv(depth)

    return sobel_x ** 2 + sobel_y ** 2


class SobelConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_func = nn.MSELoss()

    def forward(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor, threshold: float = 1.0):
        pred_sobel = get_sobel(pred_depth)
        gt_sobel = get_sobel(gt_depth)

        indices = pred_sobel < threshold

        return self.mse_func(pred_sobel[indices], gt_sobel[indices])


class SobelRegularization(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_func = nn.MSELoss()

    def forward(self, pred_depth: torch.Tensor, threshold: float = 1.0):
        pred_sobel = get_sobel(pred_depth)

        indices = pred_sobel < threshold

        return pred_sobel[indices].mean()
