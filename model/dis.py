"""
Depth Discriminator Network:
We use a discriminator referenced by WGAN-GP to make our depth prediction more realistic
"""
import torch
import torch.nn as nn


class DepthDiscriminator(nn.Module):
    def __init__(self, img_size: int = 256):
        super().__init__()

        self.img_size_after_conv = self.calculate_img_size(img_size)

        self._dis = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0)
        )
        self._linear = nn.Linear(self.img_size_after_conv ** 2, 1)

    def forward(self, x: torch.Tensor):
        b = x.size(0)

        x = self._dis(x)

        x = x.view(b, -1)
        x = self._linear(x)

        return x

    @staticmethod
    def calculate_img_size(img_size):
        def tmp(n):
            return (n - 4 + 2) // 2 + 1
        for i in range(4):
            img_size = tmp(img_size)
        return (img_size - 4) // 1 + 1

    @staticmethod
    def calculate_gradient_penalty(discriminator, real_data: torch.Tensor, fake_data: torch.Tensor):
        b = real_data.size(0)

        alpha = torch.rand((b, 1, 1, 1), requires_grad=True).cuda().expand_as(real_data)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        dis_interpolates = discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=dis_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(dis_interpolates.size()).cuda(),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0].cuda()
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
