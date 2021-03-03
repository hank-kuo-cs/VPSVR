import torch
import torch.nn as nn
from kaolin.rep import TriangleMesh
from pytorch3d.structures import Meshes
from .cd import ChamferDistanceLoss


cd_loss_func = ChamferDistanceLoss()


class ChamferNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict_meshes: list, gt_vertices: list, gt_faces: list):
        self.check_parameters(predict_meshes, gt_vertices, gt_faces)
        B = len(predict_meshes)

        edges = self.get_edges(predict_meshes[0])
        pred_vertices = torch.cat([predict_mesh.vertices[None] for predict_mesh in predict_meshes])
        pred_edges = pred_vertices[:, edges[:, 0], :] - pred_vertices[:, edges[:, 1], :]

        near_gt_indices = torch.cat([cd_loss_func(pred_vertices[b][None], gt_vertices[b][None])[1] for b in range(B)])
        gt_normals = Meshes(gt_vertices, gt_faces).verts_normals_list()  # tuple((N1, 3), (N2, 3), ... (NB, 3))

        near_gt_normals = torch.cat([normal[near_gt_indices[i], :][None] for i, normal in enumerate(gt_normals)])
        near_gt_normals = torch.nn.functional.normalize(near_gt_normals, dim=2)
        near_gt_normals = near_gt_normals[:, edges[:, 0], :]

        normal_loss = torch.abs(torch.sum(pred_edges * near_gt_normals, dim=2)).mean()
        return normal_loss

    @staticmethod
    def get_edges(predict_mesh: TriangleMesh):
        vertices, faces = predict_mesh.vertices, predict_mesh.faces

        face_size = faces.shape[1]
        edges = torch.cat([faces[:, i:i + 2] for i in range(face_size - 1)] + [faces[:, [-1, 0]]], dim=0)

        edges, _ = torch.unique(torch.sort(edges, dim=1)[0], sorted=True, return_inverse=True, dim=0)
        edges = torch.cat([edges, edges.flip(1)])

        return edges

    @staticmethod
    def check_parameters(predict_meshes: list, gt_vertices: list, gt_faces: list):
        assert isinstance(predict_meshes, list)
        assert isinstance(gt_vertices, list)
        assert isinstance(gt_faces, list)
        assert len(predict_meshes) == len(gt_vertices) == len(gt_faces)
        assert isinstance(predict_meshes[0], TriangleMesh)
