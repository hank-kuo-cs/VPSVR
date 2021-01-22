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

    # Dataset Setting
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--size', type=int, default=0, help='0 indicates all of the dataset, '
                                                            'or it will divide equally on all classes')

    # Optimizer
    parser.add_argument('--lr_den', type=float, default=1e-3, help='learning rate of depth estimation')
    parser.add_argument('--lr_vpn', type=float, default=1e-5, help='learning rate of volumetric primitives prediction')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of Adam optimizer')

    # Loss weight
    parser.add_argument('--l_depth', type=float, default=1.0, help='lambda of depth estimation loss')
    parser.add_argument('--l_vpdiv', type=float, default=0.1, help='lambda of vp diverse loss')
    parser.add_argument('--l_cd', type=float, default=1.0, help='lambda of cd loss')

    # Network
    parser.add_argument('--sphere_num', type=int, default=8, help='number of spheres')
    parser.add_argument('--cuboid_num', type=int, default=8, help='number of cuboids')

    # Training trick
    parser.add_argument('--depth_tf_ratio', type=float, default=0.0, help='teaching forcing ratio of input depth.'
                                                                          '0: not use gt depth, 1: only use gt depth')

    # Record Setting
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


from dataset import GenReDataset, R2N2Dataset, collate_func
from model import VolumetricPrimitiveNet, DepthEstimationNet
from utils.sampling import Sampling
from utils.meshing import Meshing
from utils.loss import ChamferDistanceLoss
from utils.render import DepthRenderer
from utils.visualize import save_depth_result, save_mesh_result


def load_model(args):
    den = DepthEstimationNet().cuda()
    vpn = VolumetricPrimitiveNet(vp_num=args.sphere_num + args.cuboid_num).cuda()

    return den, vpn


def set_path():
    checkpoint_paths = {'den': './checkpoint/den/', 'vpn': './checkpoint/vpn/'}
    for checkpoint_path in list(checkpoint_paths.values()):
        os.makedirs(checkpoint_path, exist_ok=True)

    record_paths = {'loss': './output/loss/train/', 'depth': './output/depth/train/', 'vp': './output/vp/train/'}
    for record_path in list(record_paths.values()):
        os.makedirs(record_path, exist_ok=True)

    return checkpoint_paths, record_paths


def train(args):
    dataset = GenReDataset(args, 'train') if args.dataset == 'genre' else R2N2Dataset(args, 'train')
    print('Load %s training dataset, size =' % args.dataset, len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=True, collate_fn=collate_func)
    checkpoint_paths, record_paths = set_path()
    den, vpn = load_model(args)
    optimizer = Adam(params=[
        {'params': den.parameters(), 'lr': args.lr_den},
        {'params': vpn.parameters(), 'lr': args.lr_vpn}
    ], betas=(args.beta1, args.beta2))

    l1_loss_func, mse_loss_func, cd_loss_func = L1Loss(), MSELoss(), ChamferDistanceLoss()

    epoch_train_losses = {'cd': [], 'vp_div': [], 'depth': []}

    for epoch in range(args.epochs):
        den.train()
        vpn.train()

        n = 0
        avg_losses = {'cd': 0.0, 'vp_div': 0.0, 'depth': 0.0}

        progress_bar = tqdm(dataloader)

        for data in progress_bar:
            # Load data
            rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
            vertices = [one_vertices.cuda() for one_vertices in data['vertices']]  # list((N1, 3), ..., (Nb, 3))
            faces = [one_faces.cuda() for one_faces in data['faces']]  # list((F1, 2), ..., (FB, 2))
            dists, elevs, azims = data['dist'].cuda(), data['elev'].cuda(), data['azim'].cuda()

            gt_depths = DepthRenderer.render_depths_of_multi_meshes(vertices, faces, normalize=True)

            # Network prediction
            predict_depths = den(rgbs)
            input_depths = predict_depths if torch.rand((1,)).item() > args.depth_tf_ratio else gt_depths

            volumes, rotates, translates, local_features, global_features = vpn(input_depths)

            # Depth loss
            depth_loss = mse_loss_func(predict_depths, gt_depths) * args.l_depth

            # VP diverse loss
            gt_meshes = Meshing.meshing_vertices_faces(vertices, faces)
            gt_points = Sampling.sample_mesh_points(gt_meshes, sample_num=2048)

            vp_center_points = torch.cat([t[:, None, :] for t in translates], 1)

            vp_div_loss = cd_loss_func(vp_center_points, gt_points, w1=0.5) * args.l_vpdiv

            # CD loss
            predict_points = Sampling.sample_vp_points(volumes, rotates, translates,
                                                       cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)

            cd_loss = cd_loss_func(predict_points, gt_points) * args.l_cd

            total_loss = vp_div_loss + cd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            progress_bar.set_description('CD loss = %.6f, depth loss = %.6f, VP diverse loss = %.6f'
                                         % (cd_loss.item(), depth_loss.item(), vp_div_loss.item()))

            avg_losses['depth'] += depth_loss.item()
            avg_losses['cd'] += cd_loss.item()
            avg_losses['vp_div'] += vp_div_loss.item()
            n += 1

            if n % args.record_batch_interval == 0:
                predict_meshes = Meshing.vp_meshing(volumes, rotates, translates,
                                                    cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)
                depth_save_path = os.path.join(record_paths['depth'], 'epoch%d-batch%d.png' % (epoch + 1, n))
                save_depth_result(rgbs[0], predict_depths[0], gt_depths[0], depth_save_path)

                mesh_save_path = os.path.join(record_paths['vp'], 'epoch%d-batch%d.png' % (epoch + 1, n))
                save_mesh_result(rgbs[0], input_depths[0],
                                 predict_meshes[0], gt_meshes[0],
                                 args.cuboid_num + args.sphere_num, mesh_save_path)

        for key in list(avg_losses.keys()):
            avg_losses[key] /= n
            epoch_train_losses[key].append(avg_losses[key])

        print('Epoch %d avg loss: CD loss = %.6f, depth loss = %.6f, VP diverse loss = %.6f\n'
              % (epoch + 1, avg_losses['cd'], avg_losses['depth'], avg_losses['vp_div']))

        # Record some result
        if (epoch+1) % args.checkpoint_epoch_interval == 0:
            torch.save(den.state_dict(), os.path.join(checkpoint_paths['den'], 'den_epoch%03d.pth' % (epoch + 1)))
            torch.save(vpn.state_dict(), os.path.join(checkpoint_paths['vpn'], 'vpn_epoch%03d.pth' % (epoch + 1)))

    for key in list(epoch_train_losses.keys()):
        np.save(os.path.join(record_paths['loss'], key + '.npy'), np.array(epoch_train_losses[key]))


if __name__ == '__main__':
    train(args)
