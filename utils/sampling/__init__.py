import torch
from kaolin.rep import TriangleMesh
from .cuboid import cuboid_sampling
from .sphere import sphere_sampling
from ..transform import transform_points


class Sampling:
    def __init__(self):
        pass

    @classmethod
    def cuboid_sampling(cls,
                        v: torch.Tensor,
                        q: torch.Tensor,
                        t: torch.Tensor,
                        num_points: int = 128):

        cls.check_parameters(v, q, t)

        canonical_points = cuboid_sampling(v, num_points)
        sample_points = transform_points(canonical_points, q, t)

        return sample_points

    @classmethod
    def sphere_sampling(cls,
                        v: torch.Tensor,
                        q: torch.Tensor,
                        t: torch.Tensor,
                        num_points: int = 128):

        cls.check_parameters(v, q, t)

        canonical_points = sphere_sampling(v, num_points)
        sample_points = transform_points(canonical_points, q, t)

        return sample_points

    @classmethod
    def cone_sampling(cls,
                      v: torch.Tensor,
                      q: torch.Tensor,
                      t: torch.Tensor,
                      num_points: int = 128):
        pass

    @staticmethod
    def check_parameters(v, q, t):
        B = v.size(0)
        assert v.size() == (B, 3)
        assert q.size() == (B, 4)
        assert t.size() == (B, 3)

    @classmethod
    def sample_vp_points(cls, volumes: list, rotates: list, translates: list,
                         cuboid_num: int, sphere_num: int, vp_sample_num: int = 128):
        vp_num = cuboid_num + sphere_num
        sample_points = []

        for i in range(vp_num):
            sampling_func = cls.cuboid_sampling if i < cuboid_num else cls.sphere_sampling
            sample_points.append(sampling_func(volumes[i], rotates[i], translates[i], vp_sample_num))

        return torch.cat(sample_points, dim=1)

    @staticmethod
    def sample_mesh_points(meshes: list, sample_num=2048):
        sample_points = []
        for mesh in meshes:
            sample_points.append(mesh.sample(sample_num)[0][None])
        return torch.cat(sample_points)
