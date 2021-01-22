import torch
from kaolin.rep import TriangleMesh
from .cuboid import cuboid_meshing
from .sphere import sphere_meshing


class Meshing:
    def __init__(self):
        pass

    @classmethod
    def vp_meshing(cls, volumes: list, rotates: list, translates: list, cuboid_num: int, sphere_num: int):
        vp_num = cuboid_num + sphere_num
        B = volumes[0].size(0)
        meshes = [[] for i in range(B)]  # (B, K)

        for i in range(vp_num):
            meshing_func = cuboid_meshing if i < cuboid_num else sphere_meshing
            vp = meshing_func(volumes[i], rotates[i], translates[i])  # (B)

            for b in range(B):
                meshes[b].append(vp[b])

        for b in range(B):
            meshes[b] = cls.compose_meshes(meshes[b])

        return meshes  # (B)

    @staticmethod
    def meshing_vertices_faces(vertices: list, faces: list):
        assert len(vertices) == len(faces)  # (B) == (B)
        meshes = []
        for b in range(len(vertices)):
            meshes.append(TriangleMesh.from_tensors(vertices[b], faces[b]))
        return meshes

    @classmethod
    def cuboid_meshing(cls, v: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> list:
        cls.check_parameters(v, q, t)
        return cuboid_meshing(v, q, t)

    @classmethod
    def sphere_meshing(cls, v: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> list:
        cls.check_parameters(v, q, t)
        return sphere_meshing(v, q, t)

    @classmethod
    def cone_meshing(cls, v: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> list:
        cls.check_parameters(v, q, t)
        pass

    @staticmethod
    def compose_meshes(meshes: list) -> TriangleMesh:
        vertices = []
        faces = []

        vertices_num = 0

        for i, mesh in enumerate(meshes):
            vertices_now = mesh.vertices.clone()
            faces_now = mesh.faces.clone()

            vertices.append(vertices_now)
            faces.append(faces_now + vertices_num)

            vertices_num += vertices_now.size(0)

        result_mesh = TriangleMesh.from_tensors(vertices=torch.cat(vertices), faces=torch.cat(faces))
        result_mesh.cuda()

        return result_mesh

    @staticmethod
    def check_parameters(v: torch.Tensor, q: torch.Tensor, t: torch.Tensor):
        assert v.size(0) == q.size(0) == t.size(0)
        B = v.size(0)

        assert v.size() == (B, 3)
        assert q.size() == (B, 4)
        assert t.size() == (B, 3)
