import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from ..perceptual_feature import get_local_features
from .bottleneck import GBottleneck, GCNConv


class DeformGCN(nn.Module):
    def __init__(self, edges: torch.Tensor, n_dim=3, feature_dim=960 + 512, v_num=2048):
        super().__init__()

        self.edges = edges
        self.blocks = nn.ModuleList([
            GBottleneck(in_dim=n_dim + feature_dim, hidden_dim=256, out_dim=256, edges=edges),
            GBottleneck(in_dim=256, hidden_dim=256, out_dim=256, edges=edges),
            GBottleneck(in_dim=256, hidden_dim=256, out_dim=256, edges=edges),
        ])
        self.last_conv = GCNConv(256, 3)

    def forward(self, meshes: list, imgs: torch.Tensor, perceptual_features: list, global_features: torch.Tensor):
        N = len(meshes[0].vertices)

        batch_vertices = self.get_batch_vertices(meshes)  # (B, N, 3)

        global_features = global_features[:, None, :].repeat(1, N, 1)
        local_features = get_local_features(batch_vertices, imgs, perceptual_features)

        x = torch.cat([batch_vertices, local_features, global_features], 2)

        deformations = self.regress(x)

        return deformations

    @staticmethod
    def get_edges(mesh: TriangleMesh):
        vertices, faces = mesh.vertices, mesh.faces

        face_size= faces.shape[1]
        edges = torch.cat([faces[:, i:i + 2] for i in range(face_size - 1)] +
                          [faces[:, [-1, 0]]], dim=0)

        edges = torch.sort(edges, dim=1)[0]
        edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
        edges = torch.cat([edges, edges.flip(1)]).view(2, -1)

        return edges

    @staticmethod
    def get_batch_vertices(meshes: list):
        return torch.cat([mesh.vertices[None] for mesh in meshes])

    def regress(self, x):
        for block in self.blocks:
            x, x_hidden = block(x)
            x = torch.cat([x, x_hidden], 1)
        x = self.last_conv(x)
        return x
