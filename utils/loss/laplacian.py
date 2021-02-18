import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from kaolin.metrics import laplacian_loss


class LaplacianRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, meshes: list, deform_meshes: list):
        self.check_parameters(meshes, deform_meshes)
        lap_loss = 0.0
        B = len(meshes)

        for b in range(B):
            lap_loss += laplacian_loss(meshes[b], deform_meshes[b])
        lap_loss /= B

        return lap_loss

    @staticmethod
    def check_parameters(meshes: list, deform_meshes: list):
        assert len(meshes) == len(deform_meshes)
        assert isinstance(meshes[0], TriangleMesh)
        assert isinstance(deform_meshes[0], TriangleMesh)
