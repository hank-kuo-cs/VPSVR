import torch
from PIL import Image
from kaolin.rep import TriangleMesh
from .image import concat_images, denormlize_image
from ..render import VertexRenderer, PhongRenderer, DepthRenderer


COLORS = torch.tensor([[0.0, 0.99, 0.0], [0.0, 0.0, 0.99], [0.99, 0.0, 0.0], [0.6, 0.3, 0.8],
                       [0.99, 0.99, 0.0], [0.0, 0.99, 0.99], [0.99, 0.0, 0.99], [0.6, 0.8, 0.3],
                       [0.3, 0.6, 0.8], [0.3, 0.8, 0.6], [0.8, 0.3, 0.6], [0.8, 0.6, 0.3],
                       [0.8, 0.1, 0.2], [0.8, 0.2, 0.1], [0.2, 0.1, 0.8], [0.2, 0.8, 0.1],
                       [0.2, 0.1, 0.3], [0.4, 0.6, 0.5], [0.9, 0.9, 0.8], [0.7, 0.99, 0.99]])


def save_vp_result(rgb: torch.Tensor, mask: torch.Tensor, input_depth: torch.Tensor,
                   predict_mesh: TriangleMesh, gt_mesh: TriangleMesh,
                   vp_num: int, save_path: str):
    assert rgb.ndimension() == 3  # (3, H, W)
    assert mask.ndimension() == 3  # (3, H, W)
    assert input_depth.ndimension() == 3  # (1, H, W)

    rgb = denormlize_image(rgb)
    rgb = rgb * mask

    vp_colors = get_vp_colors(vp_num)
    gif_imgs = []

    for azim in range(0, 360, 30):
        vp_img = VertexRenderer.render(predict_mesh, 1.0, 0.0, azim, colors=vp_colors)[0][0].permute(2, 0, 1)
        predict_depth = DepthRenderer.render_depth_of_single_mesh(predict_mesh.vertices, predict_mesh.faces, azim=azim)
        gt_depth = DepthRenderer.render_depth_of_single_mesh(gt_mesh.vertices, gt_mesh.faces, azim=azim)

        gif_imgs.append(concat_images([rgb, input_depth, vp_img, predict_depth, gt_depth]))

    gif_imgs[0].save(save_path, format='GIF', append_images=gif_imgs[1:], save_all=True, duration=300, loop=0)


def get_vp_colors(vp_num: int):
    vp_colors = []
    for i in range(vp_num):
        one_vp_color = torch.cat([torch.full(size=(128, 1), fill_value=COLORS[i, j].item()) for j in range(3)], dim=1)
        vp_colors.append(one_vp_color)

    return torch.cat(vp_colors)[None].cuda()  # (1, N, 3)


