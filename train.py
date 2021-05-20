import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from kaolin.rep import TriangleMesh


def parse_arguments():
    parser = argparse.ArgumentParser()

    # System Setting
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='training epoch num')

    # Dataset Setting
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2", "cvx_rearrange"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='root directory of dataset')
    parser.add_argument('--genre_root', type=str, default='/eva_data/hdd1/hank/GenRe', help='root directory of genre')
    parser.add_argument('--cvx_add_genre', action='store_true', default=False, help='cvx rearrangement dataset concat with genre')
    parser.add_argument('--size', type=int, default=6000, help='the size will divide equally on all classes')
    parser.add_argument('--genre_size', type=int, default=60000, help='concated genre dataset size')

    # Optimizer
    parser.add_argument('--lr_depth_ae', type=float, default=1e-3, help='learning rate of depth estimation')
    parser.add_argument('--lr_depth_en', type=float, default=1e-4, help='learning rate of depth encoder')
    parser.add_argument('--lr_translate_de', type=float, default=1e-4, help='learning rate of translate decoder')
    parser.add_argument('--lr_volume_rotate_de', type=float, default=1e-4, help='learning rate of volume rotate decoder')
    parser.add_argument('--lr_deform_de', type=float, default=1e-5, help='learning rate of deformation decoder')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of Adam optimizer')
    parser.add_argument('--w_decay', type=float, default=0.0, help='weight decay of Adam optimizer')

    # Loss weight
    parser.add_argument('--l_depth', type=float, default=1.0, help='lambda of depth mse loss')
    parser.add_argument('--l_vpdiv', type=float, default=0.5, help='lambda of vp diverse loss')
    parser.add_argument('--l_vp_cd', type=float, default=1.0, help='lambda of global vp reconstruct cd loss')
    parser.add_argument('--l_part_cd', type=float, default=0.0, help='lambda of part vp reconstruct cd loss')
    parser.add_argument('--l_deform_cd', type=float, default=1.0, help='lambda of deformed mesh reconstruct cd loss')
    parser.add_argument('--l_part_deform_cd', type=float, default=0.0, help='lambda of part deformed mesh reconstruct cd loss')
    parser.add_argument('--l_lap', type=float, default=0.0, help='lambda of laplacian regularization')
    parser.add_argument('--l_normal', type=float, default=0.0, help='lambda of normal loss')
    parser.add_argument('--l_sobel', type=float, default=0.0, help='lambda of sobel regularization loss')
    parser.add_argument('--l_vp_center', type=float, default=0.0, help='lambda of vp center loss')
    parser.add_argument('--vpdiv_w1', type=float, default=0.01, help='w1 of cd loss of vp diverse loss')

    # Volumetric Primitive
    parser.add_argument('--sphere_num', type=int, default=16, help='number of spheres')
    parser.add_argument('--cuboid_num', type=int, default=0, help='number of cuboids')
    parser.add_argument('--vertex_num', type=int, default=128, help='number of vertices of each primitive')

    # Training trick
    parser.add_argument('--dtf_decay', type=float, default=0.9, help='the decay ratio of depth teaching forcing,'
                                                                     'it will be applied every 5 epochs')

    # Record Setting
    parser.add_argument('--output_path', type=str, default='./output/train')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--record_batch_interval', type=int, default=200, help='record prediction result every N batch')
    parser.add_argument('--checkpoint_epoch_interval', type=int, default=5, help='record model checkpoint every N epoch')

    return parser.parse_args()


def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    torch.backends.cudnn.deterministic = True


args = parse_arguments()
set_seed(args.manual_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


from dataset import GenReDataset, R2N2Dataset, ConvexRearrangementDataset, collate_func
from model import DepthEstimationUNet
from model.two_step import DepthEncoder, TranslateDecoder, VolumeRotateDecoder, DeformGCN
from utils.sampling import Sampling
from utils.meshing import Meshing
from utils.loss import ChamferDistanceLoss, ChamferNormalLoss, LaplacianRegularization, SobelRegularization, VolumetricPrimitiveCenterLoss
from utils.render import DepthRenderer
from utils.perceptual import get_local_features
from utils.visualize import save_depth_result, save_vp_result


def load_model(args):
    vp_num = args.sphere_num + args.cuboid_num

    depth_ae = DepthEstimationUNet().cuda()
    depth_ae.train()

    depth_en = DepthEncoder().cuda()
    depth_en.train()

    translate_de = TranslateDecoder(vp_num=vp_num).cuda()
    translate_de.train()

    volume_rotate_de = VolumeRotateDecoder().cuda()
    volume_rotate_de.train()

    deform_gcn = DeformGCN(v_num=args.vertex_num * vp_num).cuda()

    return depth_ae, depth_en, translate_de, volume_rotate_de, deform_gcn


def set_path(args):
    network_names = ['depth_ae', 'depth_en', 'translate_de', 'volume_rotate_de', 'deform_gcn']
    checkpoint_paths = {}
    for network_name in network_names:
        checkpoint_paths[network_name] = os.path.join(args.checkpoint_path, network_name)
        os.makedirs(checkpoint_paths[network_name], exist_ok=True)

    record_names = ['loss', 'depth', 'vp', 'mesh']
    record_paths = {}
    for record_name in record_names:
        record_paths[record_name] = os.path.join(args.output_path, record_name)
        os.makedirs(record_paths[record_name], exist_ok=True)

    return checkpoint_paths, record_paths


def get_part_gt_points(gt_points: torch.Tensor, vp_indices: torch.Tensor, vp_num: int):
    B = gt_points.size(0)

    part_gt_points = [[] for i in range(vp_num)]  # len = K
    max_point_nums = [0 for i in range(vp_num)]  # len = K

    for k in range(vp_num):
        for b in range(B):
            part_gt_points_one_batch = gt_points[b, vp_indices[b] == k]
            if max_point_nums[k] < part_gt_points_one_batch.size(0):
                max_point_nums[k] = part_gt_points_one_batch.size(0)
            part_gt_points[k].append(part_gt_points_one_batch)

    for k in range(vp_num):
        for b in range(B):
            n = part_gt_points[k][b].size(0)

            deplicate_part_gt_points = torch.zeros((max_point_nums[k], 3)).cuda()
            if 0 < n < max_point_nums[k]:
                deplicate_part_gt_points[0: n, :] = part_gt_points[k][b]
                deplicate_part_gt_points[n:, :] = part_gt_points[k][b][0].expand((max_point_nums[k] - n, 3))

            part_gt_points[k][b] = deplicate_part_gt_points[None]

    for k in range(vp_num):
        part_gt_points[k] = torch.cat(part_gt_points[k])

    return part_gt_points


def train(args):
    dataset = {'genre': GenReDataset,
               '3dr2n2': R2N2Dataset,
               'cvx_rearrange': ConvexRearrangementDataset}[args.dataset](args, 'train')
    print('Load %s training dataset, size =' % args.dataset, len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=True, collate_fn=collate_func)

    vp_num = args.cuboid_num + args.sphere_num
    checkpoint_paths, record_paths = set_path(args)
    depth_tf = 1.0

    depth_ae, depth_en, translate_de, volume_rotate_de, deform_gcn = load_model(args)
    optimizer = Adam(params=[
        {'params': depth_ae.parameters(), 'lr': args.lr_depth_ae},
        {'params': depth_en.parameters(), 'lr': args.lr_depth_en},
        {'params': translate_de.parameters(), 'lr': args.lr_translate_de},
        {'params': volume_rotate_de.parameters(), 'lr': args.lr_volume_rotate_de},
        {'params': deform_gcn.parameters(), 'lr': args.lr_deform_de}
    ], betas=(args.beta1, args.beta2), weight_decay=args.w_decay)

    mse_loss_func = MSELoss()
    cd_loss_func = ChamferDistanceLoss()
    lap_loss_func = LaplacianRegularization()
    normal_loss_func = ChamferNormalLoss()
    sobel_loss_func = SobelRegularization()
    vp_center_loss_func = VolumetricPrimitiveCenterLoss()

    dists = [1. for i in range(8)]
    elevs = [0. for i in range(8)]
    azims = [45.0 * i for i in range(8)]

    epoch_train_losses = {'depth': [], 'vp_cd': [], 'vp_div': [], 'part_vp_cd': [],
                          'deform_cd': [], 'part_deform_cd': [], 'lap': [], 'normal': [], 'sobel': [], 'center': []}

    for epoch in range(args.epochs):
        n = 0
        avg_losses = {'depth': 0.0, 'vp_cd': 0.0, 'vp_div': 0.0, 'part_vp_cd': 0.0,
                      'deform_cd': 0.0, 'part_deform_cd': 0.0, 'lap': 0.0, 'normal': 0.0, 'sobel': 0.0, 'center': 0.0}

        progress_bar = tqdm(dataloader)

        for data in progress_bar:
            # Load data
            rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
            vertices = [one_vertices.cuda() for one_vertices in data['vertices']]  # list((N1, 3), ..., (Nb, 3))
            faces = [one_faces.cuda() for one_faces in data['faces']]  # list((F1, 2), ..., (FB, 2))

            gt_depths = DepthRenderer.render_depths_of_multi_meshes(vertices, faces)
            gt_meshes = Meshing.meshing_vertices_faces(vertices, faces)
            gt_points = Sampling.sample_mesh_points(gt_meshes, sample_num=8192)

            rgbs = rgbs * masks
            pred_depths = depth_ae(rgbs)
            pred_depths = pred_depths * masks

            depth_mse_loss = mse_loss_func(pred_depths, gt_depths) * args.l_depth

            # VP Diverse Loss
            input_depths = pred_depths if torch.rand((1,)).item() > depth_tf else gt_depths
            global_features, perceptual_feature_list = depth_en(input_depths)

            translates = translate_de(global_features)
            vp_centers = torch.cat([t[:, None, :] for t in translates], 1)  # (B, K, 3)
            vp_div_loss, _, vp_indices = cd_loss_func(vp_centers, gt_points, w1=args.vpdiv_w1)
            vp_div_loss *= args.l_vpdiv

            # CD loss
            volumes, rotates = [], []
            vp_features = get_local_features(vp_centers, input_depths, perceptual_feature_list)

            for i in range(vp_num):
                one_vp_feature = vp_features[:, i, :]  # (B, F)
                volume, rotate = volume_rotate_de(one_vp_feature)

                volumes.append(volume)
                rotates.append(rotate)

            pred_coarse_points = Sampling.sample_vp_points(volumes, rotates, translates,
                                                           cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)

            vp_cd_loss, _, _ = cd_loss_func(pred_coarse_points, gt_points)
            vp_cd_loss *= args.l_vp_cd

            part_point_num = pred_coarse_points.size(1) // vp_num
            part_vp_cd_loss = torch.tensor(0.0).cuda()

            part_gt_points = get_part_gt_points(gt_points, vp_indices, vp_num)

            if args.l_part_cd > 0:
                for i in range(vp_num):
                    part_pred_coarse_points = pred_coarse_points[:, i * part_point_num: (i + 1) * part_point_num, :]
                    part_vp_cd_loss += cd_loss_func(part_pred_coarse_points, part_gt_points[i])[0] * args.l_part_cd
            part_vp_cd_loss /= vp_num

            pred_meshes = Meshing.vp_meshing(volumes, rotates, translates,
                                             cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)
            vp_meshes = [TriangleMesh.from_tensors(m.vertices.clone(), m.faces.clone()) for m in pred_meshes]

            deformation = deform_gcn(pred_meshes, input_depths, perceptual_feature_list, global_features)

            for b in range(len(pred_meshes)):
                pred_meshes[b].vertices += deformation[b]

            center_loss = vp_center_loss_func(vp_num, pred_meshes, args.vertex_num, vp_centers) * args.l_vp_center \
                if args.l_vp_center > 0 else torch.tensor(0.0).cuda()

            pred_vertices = torch.cat([m.vertices[None] for m in pred_meshes])

            deform_cd_loss, _, _ = cd_loss_func(pred_vertices, gt_points)
            deform_cd_loss *= args.l_deform_cd

            part_deform_cd_loss = torch.tensor(0.0).cuda()

            if args.l_part_deform_cd > 0:
                for i in range(vp_num):
                    part_pred_vertices = pred_vertices[:, i * part_point_num: (i + 1) * part_point_num, :]
                    part_deform_cd_loss += cd_loss_func(part_pred_vertices, part_gt_points[i])[0] * args.l_part_deform_cd
            part_deform_cd_loss /= vp_num

            lap_loss = lap_loss_func(vp_meshes, pred_meshes) * args.l_lap if args.l_lap > 0 else torch.tensor(0.0).cuda()
            normal_loss = normal_loss_func(pred_meshes, vertices, faces) * args.l_normal if args.l_normal > 0 else torch.tensor(0.0).cuda()

            pred_vertices = [m.vertices for m in pred_meshes]
            pred_faces = [m.faces for m in pred_meshes]
            pred_depths = DepthRenderer.render_depths_of_multi_meshes_with_multi_view(
                pred_vertices, pred_faces, dists=dists, elevs=elevs, azims=azims)

            sobel_loss = sobel_loss_func(pred_depths) * args.l_sobel if args.l_sobel > 0 else torch.tensor(0.0).cuda()

            vp_recon_loss = vp_div_loss + vp_cd_loss + part_vp_cd_loss
            mesh_recon_loss = deform_cd_loss + part_deform_cd_loss + lap_loss + normal_loss + sobel_loss + center_loss

            total_loss = depth_mse_loss + vp_recon_loss + mesh_recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            progress_bar.set_description(
                'MSE = %.6f, VP = %.6f, Part VP = %.6f, Div = %.6f, '
                'Deform = %.6f, Part Deform = %.6g, Lap = %.6f, Normal = %.6f, Sobel = %.6f, Center = %.6f'
                % (depth_mse_loss.item(), vp_cd_loss.item(), part_vp_cd_loss.item(), vp_div_loss.item(),
                   deform_cd_loss.item(), part_deform_cd_loss.item(),
                   lap_loss.item(), normal_loss.item(), sobel_loss.item(), center_loss.item()))

            avg_losses['depth'] += depth_mse_loss.item()
            avg_losses['vp_cd'] += vp_cd_loss.item()
            avg_losses['vp_div'] += vp_div_loss.item()
            avg_losses['part_vp_cd'] += part_vp_cd_loss.item()
            avg_losses['deform_cd'] += deform_cd_loss.item()
            avg_losses['part_deform_cd'] += part_deform_cd_loss.item()
            avg_losses['lap'] += lap_loss.item()
            avg_losses['normal'] += normal_loss.item()
            avg_losses['sobel'] += sobel_loss.item()
            avg_losses['center'] += center_loss.item()
            n += 1

            if n % args.record_batch_interval == 0 and (epoch + 1) % 5 == 0:
                depth_save_path = os.path.join(record_paths['depth'], 'epoch%d-batch%d.png' % (epoch + 1, n))
                save_depth_result(rgb=rgbs[0], mask=masks[0],
                                  predict_depth=pred_depths[0], gt_depth=gt_depths[0], save_path=depth_save_path)

                vp_save_path = os.path.join(record_paths['vp'], 'epoch%d-batch%d.png' % (epoch + 1, n))
                save_vp_result(rgb=rgbs[0], mask=masks[0], input_depth=input_depths[0],
                               predict_mesh=vp_meshes[0], gt_mesh=gt_meshes[0],
                               vp_num=vp_num, vertex_num=args.vertex_num, save_path=vp_save_path)

                mesh_save_path = os.path.join(record_paths['mesh'], 'epoch%d-batch%d.png' % (epoch + 1, n))
                save_vp_result(rgb=rgbs[0], mask=masks[0], input_depth=input_depths[0],
                               predict_mesh=pred_meshes[0], gt_mesh=gt_meshes[0],
                               vp_num=vp_num, vertex_num=args.vertex_num, save_path=mesh_save_path)

        for key in list(avg_losses.keys()):
            avg_losses[key] /= n
            epoch_train_losses[key].append(avg_losses[key])

        print('Epoch %d avg loss: Depth MSE = %.6f, VP CD = %.6f, Part VP CD = %.6f, VP Div = %.6f, '
              'Deform CD = %.6f, Part Deform CD = %.6f, Lap = %.6f, Normal = %.6f, Sobel = %.6f, Center = %.6f\n'
              % (epoch + 1, avg_losses['depth'], avg_losses['vp_cd'], avg_losses['part_vp_cd'], avg_losses['vp_div'],
                 avg_losses['deform_cd'], avg_losses['part_deform_cd'], avg_losses['lap'], avg_losses['normal'],
                 avg_losses['sobel'], avg_losses['center']))

        if (epoch+1) % args.checkpoint_epoch_interval == 0:
            model_dict = {'depth_ae': depth_ae.state_dict(),
                          'depth_en': depth_en.state_dict(),
                          'translate_de': translate_de.state_dict(),
                          'volume_rotate_de': volume_rotate_de.state_dict(),
                          'deform_gcn': deform_gcn.state_dict()}

            for network_name, model_weight in model_dict.items():
                model_path = os.path.join(checkpoint_paths[network_name], '%s_epoch%03d.pth' % (network_name, epoch+1))
                torch.save(model_weight, model_path)

        depth_tf *= args.dtf_decay

    for key in list(epoch_train_losses.keys()):
        np.save(os.path.join(record_paths['loss'], key + '.npy'), np.array(epoch_train_losses[key]))


if __name__ == '__main__':
    train(args)
