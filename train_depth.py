import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss
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
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2" or "cvx_rearrange"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--genre_root', type=str, default='/eva_data/hdd1/hank/GenRe', help='root directory of genre')
    parser.add_argument('--cvx_add_genre', action='store_true', help='cvx rearrangement dataset concat with genre')
    parser.add_argument('--size', type=int, default=0, help='the size will divide equally on all classes')
    parser.add_argument('--genre_size', type=int, default=60000, help='concated genre dataset size')

    # Optimizer
    parser.add_argument('--lr_den', type=float, default=1e-3, help='learning rate of depth estimation')
    parser.add_argument('--lr_dis', type=float, default=1e-5, help='learning rate of depth discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay of Adam optimizer')

    # Loss weight
    parser.add_argument('--l_depth', type=float, default=1.0, help='lambda of depth estimation loss')
    parser.add_argument('--l_dis', type=float, default=1.0, help='lambda of depth discriminate loss')
    parser.add_argument('--l_gp', type=float, default=0.1, help='lambda of gradient penalty of discriminator')

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


from dataset import GenReDataset, R2N2Dataset, ConvexRearrangementDataset, collate_func
from model import DepthEstimationUNet, DepthDiscriminator
from utils.render import DepthRenderer
from utils.visualize import save_depth_result


def set_path():
    checkpoint_paths = {'den': './checkpoint/den/', 'dis': './checkpoint/dis/'}
    for checkpoint_path in list(checkpoint_paths.values()):
        os.makedirs(checkpoint_path, exist_ok=True)

    record_paths = {'loss': './output/loss/train/', 'depth': './output/depth/train/'}
    for record_path in list(record_paths.values()):
        os.makedirs(record_path, exist_ok=True)

    return checkpoint_paths, record_paths


def train(args):
    dataset = {'genre': GenReDataset(args, 'train'),
               '3dr2n2': R2N2Dataset(args, 'train'),
               'cvx_rearrange': ConvexRearrangementDataset(args)}[args.dataset]
    print('Load %s training dataset, size =' % args.dataset, len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=True, collate_fn=collate_func)
    checkpoint_paths, record_paths = set_path()
    den = DepthEstimationUNet().cuda()
    dis = DepthDiscriminator().cuda()

    optimizer_den = Adam(params=den.parameters(), lr=args.lr_den,
                         betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    optimizer_dis = Adam(params=dis.parameters(), lr=args.lr_dis,
                         betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    mse_loss_func = MSELoss()
    epoch_losses = {'depth': [], 'dis_real': [], 'dis_fake': [], 'den_real': []}

    for epoch in range(args.epochs):
        den.train()
        dis.train()

        n = 0
        avg_losses = {'depth': 0.0, 'dis_real': 0.0, 'dis_fake': 0.0, 'den_real': 0.0}

        progress_bar = tqdm(dataloader)

        for data in progress_bar:
            # Load data
            rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
            vertices = [one_vertices.cuda() for one_vertices in data['vertices']]  # list((N1, 3), ..., (Nb, 3))
            faces = [one_faces.cuda() for one_faces in data['faces']]  # list((F1, 2), ..., (FB, 2))

            # Predict Depth
            gt_depths = DepthRenderer.render_depths_of_multi_meshes(vertices, faces)
            predict_depths = den(rgbs)
            predict_depths_detach = predict_depths.clone().detach()

            # Update Depth Discriminator Network
            optimizer_dis.zero_grad()

            dis_real_loss = -dis(gt_depths).mean() * args.l_dis
            dis_fake_loss = dis(predict_depths_detach).mean() * args.l_dis

            gp = DepthDiscriminator.calculate_gradient_penalty(dis, gt_depths, predict_depths_detach) * args.l_gp

            dis_loss = dis_real_loss + dis_fake_loss + gp
            dis_loss.backward()
            optimizer_dis.step()

            # Update Depth Estimation Network
            optimizer_den.zero_grad()

            den_real_loss = -dis(predict_depths).mean() * args.l_dis

            depth_loss = mse_loss_func(predict_depths, gt_depths) * args.l_depth

            den_loss = den_real_loss + depth_loss
            den_loss.backward()
            optimizer_den.step()

            progress_bar.set_description('depth loss = %.6f, dis real = %.6f, dis fake = %.6f, den real = %.6f'
                                         % (depth_loss.item(), dis_real_loss.item(),
                                            dis_fake_loss.item(), den_real_loss.item()))

            avg_losses['depth'] += depth_loss.item()
            avg_losses['dis_real'] += dis_real_loss.item()
            avg_losses['dis_fake'] += dis_fake_loss.item()
            avg_losses['den_real'] += den_real_loss.item()
            n += 1

            if n % args.record_batch_interval == 0:
                depth_save_path = os.path.join(record_paths['depth'], 'epoch%d-batch%d.png' % (epoch + 1, n))
                save_depth_result(rgbs[0], predict_depths[0], gt_depths[0], depth_save_path)

        for k in list(avg_losses.keys()):
            avg_losses[k] /= n
            epoch_losses[k].append(avg_losses[k])

        print('Epoch %d avg loss: depth loss = %.6f, dis real = %.6f, dis fake = %.6f, den real = %.6f\n'
              % (epoch+1, avg_losses['depth'], avg_losses['dis_real'], avg_losses['dis_fake'], avg_losses['den_real']))

        if (epoch + 1) % args.checkpoint_epoch_interval == 0:
            torch.save(den.state_dict(), os.path.join(checkpoint_paths['den'], 'den_epoch%03d.pth' % (epoch + 1)))
            torch.save(dis.state_dict(), os.path.join(checkpoint_paths['dis'], 'dis_epoch%03d.pth' % (epoch + 1)))

    for k in list(epoch_losses.keys()):
        np.save(os.path.join(record_paths['loss'], '%s.npy' % str(k)), np.array(epoch_losses[k]))


if __name__ == '__main__':
    train(args)
