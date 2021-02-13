import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser()

    # System Setting
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='training epoch num')
    parser.add_argument('--use_symmetry', action='store_true', help='whether use symmetry features fusion')

    # Dataset Setting
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2", "cvx_rearrange"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='root directory of dataset')
    parser.add_argument('--genre_root', type=str, default='/eva_data/hdd1/hank/GenRe', help='root directory of genre')
    parser.add_argument('--cvx_add_genre', action='store_true', help='cvx rearrangement dataset concat with genre')
    parser.add_argument('--depth_unet_path', type=str, default='checkpoint/depth_unet.pth')
    parser.add_argument('--size', type=int, default=60000, help='the size will divide equally on all classes')
    parser.add_argument('--genre_size', type=int, default=60000, help='concated genre dataset size')

    # Optimizer
    parser.add_argument('--lr_depth_en', type=float, default=1e-5, help='learning rate of depth encoder')
    parser.add_argument('--lr_translate_de', type=float, default=1e-5, help='learning rate of translate decoder')
    parser.add_argument('--lr_volume_rotate_de', type=float, default=1e-5, help='learning rate of volume rotate decoder')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of Adam optimizer')

    # Loss weight
    parser.add_argument('--l_vpdiv', type=float, default=0.5, help='lambda of vp diverse loss')
    parser.add_argument('--l_cd', type=float, default=1.0, help='lambda of obj reconstruct cd loss')
    parser.add_argument('--l_part_cd', type=float, default=1.0, help='lambda of part reconstruct cd loss')
    parser.add_argument('--vpdiv_w1', type=float, default=0.01, help='w1 of cd loss of vp diverse loss')

    # Volumetric Primitive
    parser.add_argument('--sphere_num', type=int, default=8, help='number of spheres')
    parser.add_argument('--cuboid_num', type=int, default=8, help='number of cuboids')

    # Training trick
    parser.add_argument('--depth_tf_ratio', type=float, default=0.0, help='teaching forcing ratio of input depth.'
                                                                          '0: not use gt depth, 1: only use gt depth')

    # Record Setting
    parser.add_argument('--output_path', type=str, default='./output/train')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--record_batch_interval', type=int, default=100, help='record prediction result every N batch')
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
from model.two_step import DepthEncoder, TranslateDecoder, VolumeRotateDecoder
from utils.sampling import Sampling
from utils.meshing import Meshing
from utils.loss import ChamferDistanceLoss
from utils.render import DepthRenderer
from utils.perceptual import get_local_features
from utils.transform import get_symmetrical_points
from utils.visualize import save_depth_result, save_mesh_result


def load_model(args):
    depth_unet = DepthEstimationUNet().cuda()
    depth_unet.load_state_dict(torch.load(args.depth_unet_path))

    depth_en = DepthEncoder().cuda()

    translate_de = TranslateDecoder(vp_num=args.cuboid_num + args.sphere_num).cuda()

    local_feature_dim = 960 * 2 if args.use_symmetry else 960
    volume_rotate_de = VolumeRotateDecoder(feature_dim=local_feature_dim).cuda()

    return depth_unet, depth_en, translate_de, volume_rotate_de


def set_path(args):
    network_names = ['depth_en', 'translate_de', 'volume_rotate_de']
    checkpoint_paths = {}
    for network_name in network_names:
        checkpoint_paths[network_name] = os.path.join(args.checkpoint_path, network_name)
        os.makedirs(checkpoint_paths[network_name], exist_ok=True)

    record_names = ['loss', 'depth', 'vp']
    record_paths = {}
    for record_name in record_names:
        record_paths[record_name] = os.path.join(args.output_path, record_name)
        os.makedirs(record_paths[record_name], exist_ok=True)

    return checkpoint_paths, record_paths


def get_vp_features(vp_centers: torch.Tensor, imgs: torch.Tensor, perceptual_features_list: list,
                    dist: torch.Tensor, elev: torch.Tensor, azim: torch.Tensor, use_symmetry: bool):
    vp_local_features = get_local_features(vp_centers, imgs, perceptual_features_list)
    if not use_symmetry:
        return vp_local_features

    symmetric_points = get_symmetrical_points(vp_centers, dist, elev, azim)
    sym_local_features = get_local_features(symmetric_points, imgs, perceptual_features_list)

    return torch.cat([vp_local_features, sym_local_features], -1)


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
    depth_unet, depth_en, translate_de, volume_rotate_de = load_model(args)
    optimizer = Adam(params=[
        {'params': depth_en.parameters(), 'lr': args.lr_depth_en},
        {'params': translate_de.parameters(), 'lr': args.lr_translate_de},
        {'params': volume_rotate_de.parameters(), 'lr': args.lr_volume_rotate_de},
    ], betas=(args.beta1, args.beta2))

    cd_loss_func = ChamferDistanceLoss()

    epoch_train_losses = {'obj_cd': [], 'vp_div': [], 'part_cd': []}

    depth_unet.eval()
    depth_en.train()
    translate_de.train()
    volume_rotate_de.train()

    for epoch in range(args.epochs):
        n = 0
        avg_losses = {'obj_cd': 0.0, 'vp_div': 0.0, 'part_cd': 0.0}

        progress_bar = tqdm(dataloader)

        for data in progress_bar:
            # Load data
            rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
            vertices = [one_vertices.cuda() for one_vertices in data['vertices']]  # list((N1, 3), ..., (Nb, 3))
            faces = [one_faces.cuda() for one_faces in data['faces']]  # list((F1, 2), ..., (FB, 2))
            dists, elevs, azims = data['dist'].cuda(), data['elev'].cuda(), data['azim'].cuda()

            gt_depths = DepthRenderer.render_depths_of_multi_meshes(vertices, faces, normalize=True)
            gt_meshes = Meshing.meshing_vertices_faces(vertices, faces)
            gt_points = Sampling.sample_mesh_points(gt_meshes, sample_num=2048)

            # rgbs = rgbs * masks
            with torch.no_grad():
                predict_depths = depth_unet(rgbs)
                predict_depths = predict_depths * masks

            # VP Diverse Loss
            input_depths = predict_depths if torch.rand((1,)).item() > args.depth_tf_ratio else gt_depths
            global_features, perceptual_feature_list = depth_en(input_depths)

            translates = translate_de(global_features)
            vp_centers = torch.cat([t[:, None, :] for t in translates], 1)  # (B, K, 3)
            vp_div_loss, _, vp_indices = cd_loss_func(vp_centers, gt_points, w1=args.vpdiv_w1)
            vp_div_loss *= args.l_vpdiv

            # CD loss
            volumes, rotates = [], []
            vp_features = get_vp_features(vp_centers, input_depths, perceptual_feature_list,  # (B, K, F)
                                          dists, elevs, azims, use_symmetry=args.use_symmetry)
            for i in range(vp_num):
                one_vp_feature = vp_features[:, i, :]  # (B, F)
                volume, rotate = volume_rotate_de(one_vp_feature)
                volumes.append(volume)
                rotates.append(rotate)

            predict_points = Sampling.sample_vp_points(volumes, rotates, translates,
                                                       cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)

            cd_loss, _, _ = cd_loss_func(predict_points, gt_points)
            cd_loss *= args.l_cd

            part_point_num = predict_points.size(1) // vp_num
            part_cd_loss = 0.0

            part_gt_points = get_part_gt_points(gt_points, vp_indices, vp_num)

            for i in range(vp_num):
                part_cd_loss += cd_loss_func(predict_points[:, i*part_point_num: (i+1)*part_point_num, :],
                                             part_gt_points[i])[0] * args.l_part_cd

            total_loss = vp_div_loss + cd_loss + part_cd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            progress_bar.set_description('Obj CD Loss = %.6f, Part CD Loss = %.6f, VP Div Loss = %.6f'
                                         % (cd_loss.item(), part_cd_loss.item(), vp_div_loss.item()))

            avg_losses['obj_cd'] += cd_loss.item()
            avg_losses['vp_div'] += vp_div_loss.item()
            avg_losses['part_cd'] += part_cd_loss.item()
            n += 1

            if n % args.record_batch_interval == 0 and (epoch + 1) % 5 == 0:
                predict_meshes = Meshing.vp_meshing(volumes, rotates, translates,
                                                    cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)
                depth_save_path = os.path.join(record_paths['depth'], 'epoch%d-batch%d.png' % (epoch + 1, n))
                save_depth_result(rgbs[0], predict_depths[0], gt_depths[0], depth_save_path)

                mesh_save_path = os.path.join(record_paths['vp'], 'epoch%d-batch%d.png' % (epoch + 1, n))
                save_mesh_result(rgbs[0], input_depths[0], predict_meshes[0], gt_meshes[0], vp_num, mesh_save_path)

        for key in list(avg_losses.keys()):
            avg_losses[key] /= n
            epoch_train_losses[key].append(avg_losses[key])

        print('Epoch %d avg loss: Obj CD Loss = %.6f, Part CD Loss = %.6f, VP Div Loss = %.6f\n'
              % (epoch + 1, avg_losses['obj_cd'], avg_losses['part_cd'], avg_losses['vp_div']))

        # Record some result
        if (epoch+1) % args.checkpoint_epoch_interval == 0:
            model_dict = {'depth_en': depth_en.state_dict(),
                          'translate_de': translate_de.state_dict(),
                          'volume_rotate_de': volume_rotate_de.state_dict()}

            for network_name, model_weight in model_dict.items():
                model_path = os.path.join(checkpoint_paths[network_name], '%s_epoch%03d.pth' % (network_name, epoch+1))
                torch.save(model_weight, model_path)

    for key in list(epoch_train_losses.keys()):
        np.save(os.path.join(record_paths['loss'], key + '.npy'), np.array(epoch_train_losses[key]))


if __name__ == '__main__':
    train(args)
