import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import MSELoss
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser()

    # System Setting
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Evaluation Setting
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--depth_unet_path', type=str, default='checkpoint/depth_unet/depth_unet_epoch050.pth')
    parser.add_argument('--depth_en_path', type=str, default='checkpoint/depth_en/depth_en_epoch050.pth')
    parser.add_argument('--translate_de_path', type=str, default='checkpoint/translate_de/translate_de_epoch050.pth')
    parser.add_argument('--volume_rotate_de_path', type=str, default='checkpoint/volume_rotate_de/volume_rotate_de_epoch050.pth')
    parser.add_argument('--use_symmetry', action='store_true', help='whether use symmetry features fusion')
    parser.add_argument('--use_gt_depth', action='store_true', help='whether use gt depth as network input')

    # Dataset Setting
    parser.add_argument('--unseen', action='store_true', default=True, help='eval on unseen or seen classes')
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--size', type=int, default=0, help='0 indicates all of the dataset, '
                                                            'or it will divide equally on all classes')

    # Volumetric Primitive
    parser.add_argument('--sphere_num', type=int, default=8, help='number of spheres')
    parser.add_argument('--cuboid_num', type=int, default=8, help='number of cuboids')

    # Training trick
    parser.add_argument('--depth_tf_ratio', type=float, default=0.0, help='teaching forcing ratio of input depth.'
                                                                          '0: not use gt depth, 1: only use gt depth')

    # Record Setting
    parser.add_argument('--output_path', type=str, default='./output/eval/epoch50')
    parser.add_argument('--record_batch_interval', type=int, default=20, help='record prediction result every N batch')

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


from dataset import GenReDataset, R2N2Dataset, collate_func, CLASS_DICT
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
    depth_en.load_state_dict(torch.load(args.depth_en_path))

    translate_de = TranslateDecoder(vp_num=args.cuboid_num + args.sphere_num).cuda()
    translate_de.load_state_dict(torch.load(args.translate_de_path))

    global_feature_dim = 512
    local_feature_dim = 960 * 2 if args.use_symmetry else 960
    volume_rotate_de = VolumeRotateDecoder(feature_dim=global_feature_dim + local_feature_dim).cuda()
    volume_rotate_de.load_state_dict(torch.load(args.volume_rotate_de_path))

    return depth_unet, depth_en, translate_de, volume_rotate_de


def set_path(args):
    record_paths = {'loss': os.path.join(args.output_path, 'loss'),
                    'depth': os.path.join(args.output_path, 'depth'),
                    'vp': os.path.join(args.output_path, 'vp')}
    for record_path in list(record_paths.values()):
        os.makedirs(record_path, exist_ok=True)

    return record_paths


def get_vp_features(vp_centers: torch.Tensor, imgs: torch.Tensor, perceptual_features_list: list,
                    dist: torch.Tensor, elev: torch.Tensor, azim: torch.Tensor, use_symmetry: bool):
    vp_local_features = get_local_features(vp_centers, imgs, perceptual_features_list)
    if not use_symmetry:
        return vp_local_features

    symmetric_points = get_symmetrical_points(vp_centers, dist, elev, azim)
    sym_local_features = get_local_features(symmetric_points, imgs, perceptual_features_list)

    return torch.cat([vp_local_features, sym_local_features], -1)


@torch.no_grad()
def eval(args):
    dataset = GenReDataset(args, 'test') if args.dataset == 'genre' else R2N2Dataset(args, 'test')
    print('Load %s testing dataset, size =' % args.dataset, len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=False, collate_fn=collate_func)

    vp_num = args.cuboid_num + args.sphere_num
    record_paths = set_path(args)
    depth_unet, depth_en, translate_de, volume_rotate_de = load_model(args)

    depth_unet.eval()
    depth_en.eval()
    translate_de.eval()
    volume_rotate_de.eval()

    mse_loss_func, cd_loss_func = MSELoss(), ChamferDistanceLoss()

    class_losses = {'depth': {}, 'cd': {}}
    class_n = {}

    progress_bar = tqdm(enumerate(dataloader))

    for idx, data in progress_bar:
        rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
        class_ids = data['class_id']
        vertices = [one_vertices.cuda() for one_vertices in data['vertices']]  # list((N1, 3), ..., (Nb, 3))
        faces = [one_faces.cuda() for one_faces in data['faces']]
        dists, elevs, azims = data['dist'].cuda(), data['elev'].cuda(), data['azim'].cuda()

        gt_depths = DepthRenderer.render_depths_of_multi_meshes(vertices, faces, normalize=True)
        gt_meshes = Meshing.meshing_vertices_faces(vertices, faces)
        gt_points = Sampling.sample_mesh_points(gt_meshes, sample_num=1024)

        rgbs = rgbs * masks
        predict_depths = depth_unet(rgbs)

        input_depths = gt_depths if args.use_gt_depth else predict_depths

        global_features, perceptual_feature_list = depth_en(input_depths)

        translates = translate_de(global_features)
        vp_center_points = torch.cat([t[:, None, :] for t in translates], 1)  # (B, K, 3)

        volumes, rotates = [], []
        vp_features = get_vp_features(vp_center_points, input_depths, perceptual_feature_list,  # (B, K, F)
                                      dists, elevs, azims, use_symmetry=args.use_symmetry)
        for i in range(vp_num):
            one_vp_feature = vp_features[:, i, :]  # (B, F)
            volume, rotate = volume_rotate_de(global_features, one_vp_feature)
            volumes.append(volume)
            rotates.append(rotate)

        predict_meshes = Meshing.vp_meshing(volumes, rotates, translates,
                                            cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)
        predict_points = Sampling.sample_mesh_points(predict_meshes, sample_num=1024)

        cd_loss = cd_loss_func(predict_points, gt_points, each_batch=True)

        batch_size = rgbs.size(0)
        for b in range(batch_size):
            depth_loss = mse_loss_func(predict_depths[b][None], gt_depths[b][None])
            class_id = class_ids[b]

            if class_id in class_losses['depth']:
                class_losses['depth'][class_id] += depth_loss
                class_losses['cd'][class_id] += cd_loss[b]
                class_n[class_id] += 1
            else:
                class_losses['depth'][class_id] = depth_loss
                class_losses['cd'][class_id] = cd_loss[b]
                class_n[class_id] = 1

        if (idx + 1) % args.record_batch_interval == 0:
            depth_save_path = os.path.join(record_paths['depth'], 'batch%d.png' % (idx + 1))
            vp_save_path = os.path.join(record_paths['vp'], 'batch%d.png' % (idx + 1))

            save_depth_result(rgbs[0], predict_depths[0], gt_depths[0], depth_save_path)
            save_mesh_result(rgbs[0], input_depths[0],
                             predict_meshes[0], gt_meshes[0],
                             args.cuboid_num + args.sphere_num, vp_save_path)

    avg_depth_loss, avg_cd_loss = 0.0, 0.0

    print('Depth MSE Loss')
    print('id\t\tloss\t\tname')

    for k in list(class_losses['depth'].keys()):
        class_losses['depth'][k] = (class_losses['depth'][k] / class_n[k]).item()
        print('%s\t%.6f\t%s' % (k, class_losses['depth'][k], CLASS_DICT[k]))
        avg_depth_loss += class_losses['depth'][k]

    avg_depth_loss /= len(list(class_losses['depth'].keys()))
    print('total mean depth mse loss = %.6f' % avg_depth_loss)
    class_losses['depth']['total'] = avg_depth_loss

    np.savez(os.path.join(record_paths['loss'], 'depth.npz'), **class_losses['depth'])

    print('Mesh CD Loss')
    print('id\t\tloss\t\tname')

    for k in list(class_losses['cd'].keys()):
        class_losses['cd'][k] = (class_losses['cd'][k] / class_n[k]).item()
        print('%s\t%.6f\t%s' % (k, class_losses['cd'][k], CLASS_DICT[k]))
        avg_cd_loss += class_losses['cd'][k]

    avg_cd_loss /= len(list(class_losses['cd'].keys()))
    print('total mean mesh cd loss = %.6f' % avg_cd_loss)
    class_losses['cd']['total'] = avg_cd_loss

    np.savez(os.path.join(record_paths['loss'], 'cd.npz'), **class_losses['cd'])


if __name__ == '__main__':
    eval(args)
