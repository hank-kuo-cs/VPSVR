import os
import random
import kaolin
import numpy as np
import argparse
import torch
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    # System Setting
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='training epoch num')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of camera estimation')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of Adam optimizer')

    # Dataset Setting
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--size', type=int, default=0, help='0 indicates all of the dataset, '
                                                            'or it will divide equally on all classes')

    # Record Setting
    parser.add_argument('--output_path', type=str, default='./output/train')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--checkpoint_epoch_interval', type=int, default=5,
                        help='record model checkpoint every N epoch')

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


def set_path(args):
    network_names = ['cen']
    checkpoint_paths = {}
    for network_name in network_names:
        checkpoint_paths[network_name] = os.path.join(args.checkpoint_path, network_name)
        os.makedirs(checkpoint_paths[network_name], exist_ok=True)

    record_names = ['loss']
    record_paths = {}
    for record_name in record_names:
        record_paths[record_name] = os.path.join(args.output_path, record_name)
        os.makedirs(record_paths[record_name], exist_ok=True)

    return checkpoint_paths, record_paths


from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import GenReDataset, R2N2Dataset, collate_func
from model import CameraEstimationNetwork
from utils.loss import SphericalCoordinateMSE


def train():
    dataset = GenReDataset(args, 'train') if args.dataset == 'genre' else R2N2Dataset(args, 'train')
    print('Load %s training dataset, size =' % args.dataset, len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=True, collate_fn=collate_func)

    checkpoint_paths, record_paths = set_path(args)

    cen = CameraEstimationNetwork().cuda()
    optimizer = Adam(params=cen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scmse_func = SphericalCoordinateMSE()

    for epoch in range(args.epochs):
        cen.train()

        avg_loss = 0.0
        n = 0

        for data in tqdm(dataloader):
            rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
            dists, elevs, azims = data['dist'].cuda(), data['elev'].cuda(), data['azim'].cuda()

            elevs = elevs / 90
            azims = azims / 360

            pred_elevs, pred_azims = cen(rgbs)
            sc_mse_loss = scmse_func(pred_elevs, pred_azims, elevs, azims)

            optimizer.zero_grad()
            sc_mse_loss.backward()
            optimizer.step()

            avg_loss += sc_mse_loss.item()
            n += 1

        avg_loss /= n
        print('Epoch %d, mse loss = %.6f\n' % (epoch + 1, avg_loss))

        if (epoch+1) % args.checkpoint_epoch_interval == 0:
            model_path = os.path.join(checkpoint_paths['cen'], 'cen_epoch%03d.pth' % (epoch+1))
            torch.save(cen.state_dict(), model_path)


if __name__ == '__main__':
    train()
