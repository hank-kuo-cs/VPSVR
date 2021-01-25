import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from model import DepthEstimationUNet, VolumetricPrimitiveNet


def parse_arguments():
    parser = argparse.ArgumentParser()

    # System Setting
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Evaluate Setting
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--den_path', type=str, default='checkpoint/den/den_epoch050.pth')
    parser.add_argument('--vpn_path', type=str, default='checkpoint/vpn/vpn_epoch050.pth')
    parser.add_argument('--use_gt_depth', action='store_true')

    # Dataset Setting
    parser.add_argument('--unseen', action='store_true', default=True, help='eval on unseen or seen classes')
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--size', type=int, default=0, help='0 indicates all of the dataset, '
                                                            'or it will divide equally on all classes')
    # Volumetric Primitive
    parser.add_argument('--sphere_num', type=int, default=8, help='number of spheres')
    parser.add_argument('--cuboid_num', type=int, default=8, help='number of cuboids')

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


from torch.utils.data import DataLoader
from torch.nn import MSELoss
from dataset import GenReDataset, R2N2Dataset, collate_func, CLASS_DICT
from utils.meshing import Meshing
from utils.sampling import Sampling
from utils.loss import ChamferDistanceLoss
from utils.render import DepthRenderer
from utils.visualize import save_mesh_result, save_depth_result


def load_model(args):
    den = DepthEstimationUNet().cuda()
    den.load_state_dict(torch.load(args.den_path))

    vpn = VolumetricPrimitiveNet(vp_num=args.sphere_num + args.cuboid_num).cuda()
    vpn.load_state_dict(torch.load(args.vpn_path))

    return den, vpn


def set_path(args):
    record_paths = {'loss': os.path.join(args.output_path, 'loss'),
                    'depth': os.path.join(args.output_path, 'depth'),
                    'vp': os.path.join(args.output_path, 'vp')}
    for record_path in list(record_paths.values()):
        os.makedirs(record_path, exist_ok=True)

    return record_paths


@torch.no_grad()
def eval(args):
    dataset = GenReDataset(args, 'test') if args.dataset == 'genre' else R2N2Dataset(args, 'test')
    print('Load %s testing dataset, size =' % args.dataset, len(dataset))

    den, vpn = load_model(args)
    den.eval()
    vpn.eval()

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=False, collate_fn=collate_func)

    record_paths = set_path(args)

    mse_loss_func = MSELoss()
    cd_loss_func = ChamferDistanceLoss()
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

        rgbs = rgbs * masks
        predict_depths = den(rgbs)
        predict_depths = predict_depths * masks

        input_depths = gt_depths if args.use_gt_depth else predict_depths

        volumes, rotates, translates, local_features, global_features = vpn(input_depths)

        gt_meshes = Meshing.meshing_vertices_faces(vertices, faces)
        gt_points = Sampling.sample_mesh_points(gt_meshes, sample_num=1024)

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
