import torch
import torch.nn as nn


class VolumetricPrimitiveCenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vp_num: int, pred_meshes: list, vertex_num: int, vp_centers: torch.Tensor):
        part_centers = []
        for i in range(vp_num):
            part_vertices = torch.cat([m.vertices[i * vertex_num: (i + 1) * vertex_num, :][None] for m in pred_meshes])
            part_centers.append(part_vertices.mean(1)[:, None, :])
        part_centers = torch.cat(part_centers, 1)

        center_loss = torch.norm(part_centers - vp_centers.clone().detach(), dim=1).mean()

        return center_loss
