import torch
import random
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, RasterizationSettings, OpenGLPerspectiveCameras, MeshRasterizer


class DepthRenderer:
    def __init__(self):
        pass

    @classmethod
    def render_depths_of_multi_meshes_with_multi_view(cls, multi_vertices: list, multi_faces: list,
                                                      dists: list, elevs: list, azims: list,
                                                      img_size=256, normalize=True, device='cuda'):
        assert len(dists) == len(elevs) == len(azims)
        view_num = len(dists)
        multi_depths = []

        for i in range(view_num):
            rendered_depths = cls.render_depths_of_multi_meshes(multi_vertices, multi_faces,
                                                                dist=dists[i], elev=elevs[i], azim=azims[i],
                                                                img_size=img_size, normalize=normalize, device=device)
            multi_depths.append(rendered_depths)

        return torch.cat(multi_depths, 1)  # (B, N, H, W)

    @classmethod
    def render_depths_of_single_mesh_with_multi_view(cls, vertices: torch.Tensor, faces: torch.Tensor,
                                                     dists: list, elevs: list, azims: list,
                                                     img_size=256, normalize=True, device='cuda'):
        assert len(dists) == len(elevs) == len(azims)
        view_num = len(dists)
        multi_depth = []

        for i in range(view_num):
            rendered_depth = cls.render_depth_of_single_mesh(vertices, faces,
                                                             dist=dists[i], elev=elevs[i], azim=azims[i],
                                                             img_size=img_size, normalize=normalize, device=device)
            multi_depth.append(rendered_depth)

        return torch.cat(multi_depth, 0)  # (N, H, W)

    @classmethod
    def render_depths_of_multi_meshes(cls, multi_vertices: list, multi_faces: list,
                                      dist=1.0, elev=0.0, azim=0.0,
                                      img_size=256, normalize=True, device='cuda'):
        assert len(multi_vertices) == len(multi_faces)

        meshes = Meshes(verts=multi_vertices, faces=multi_faces)
        azim = 90 - azim  # Pytorch3D axis is different from ShapeNet and Kaolin

        # camera setting
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        cameras = OpenGLPerspectiveCameras(fov=50, R=R, T=T, device=device)

        # rasterizer setting
        raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        rendered_depths = rasterizer(meshes).zbuf.permute(0, 3, 1, 2)  # (B, 1, H, W)

        if normalize:
            for i, rendered_depth in enumerate(rendered_depths):
                rendered_depths[i] = cls.normalize_depth(rendered_depth)

        return rendered_depths  # (B, 1, H, W)

    @classmethod
    def render_depth_of_single_mesh(cls, vertices: torch.Tensor, faces: torch.Tensor,
                                    dist=1.0, elev=0.0, azim=0.0,
                                    img_size=256, normalize=True, device='cuda'):
        meshes = Meshes(verts=[vertices], faces=[faces])
        azim = 90 - azim  # Pytorch3D axis is different from ShapeNet and Kaolin

        # camera setting
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        cameras = OpenGLPerspectiveCameras(fov=50, R=R, T=T, device=device)

        # rasterizer setting
        raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        # render depth
        rendered_depth = rasterizer(meshes).zbuf[0].permute(2, 0, 1)  # (1, H, W)

        return rendered_depth if not normalize else cls.normalize_depth(rendered_depth)  # (1, H, W)

    @staticmethod
    def normalize_depth(depth: torch.Tensor):
        assert depth.ndimension() == 3  # (1, H, W)
        assert depth.size(0) == 1

        depth_indices = depth >= 0
        non_depth_indices = depth < 0

        depth[depth_indices] = depth[depth_indices].max() - depth[depth_indices]
        depth[depth_indices] /= depth[depth_indices].max()
        depth[non_depth_indices] = torch.zeros_like(depth[non_depth_indices])

        return depth

    @staticmethod
    def get_random_view_poses(view_num: int,
                              dist_low=1.0, dist_high=1.0, elev_low=-90, elev_high=90, azim_low=0, azim_high=360):
        dists = dist_low + (dist_high - dist_low) * torch.rand(view_num)
        elevs = elev_low + (elev_high - elev_low) * torch.rand(view_num)
        azims = azim_low + (azim_high - azim_low) * torch.rand(view_num)

        return dists.tolist(), elevs.tolist(), azims.tolist()
