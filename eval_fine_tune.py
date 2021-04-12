import os
import torch
import random
import argparse
import numpy as np
from kaolin.rep import TriangleMesh
from tqdm import tqdm
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.optim import Adam


def parse_arguments():
    parser = argparse.ArgumentParser()

    # System
    parser.add_argument('--gpu', type=str, default='0', help='device number of gpu')
    parser.add_argument('--manual_seed', type=int, default=0, help='manual seed for randomness')

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=20, help='use which epoch to test,'
                                                              'it will be ignore if below model paths given')
    parser.add_argument('--depth_ae_path', type=str, default='')
    parser.add_argument('--depth_en_path', type=str, default='')
    parser.add_argument('--translate_de_path', type=str, default='')
    parser.add_argument('--volume_rotate_de_path', type=str, default='')
    parser.add_argument('--deform_de_path', type=str, default='')
    parser.add_argument('--use_gt_depth', action='store_true', help='whether use gt depth as network input')

    # Dataset
    parser.add_argument('--dataset', type=str, default='genre', help='choose "genre" or "3dr2n2"')
    parser.add_argument('--root', type=str, default='/eva_data/hdd1/hank/GenRe', help='the root directory of dataset')
    parser.add_argument('--size', type=int, default=0, help='0 indicates all of the dataset, '
                                                            'or it will divide equally on all classes')
    # Fine Tune
    parser.add_argument('--lr', type=float, default=1e-5, help='lr of optimizer')
    parser.add_argument('--iter_num', type=int, default=20, help='number of iterations')
    parser.add_argument('--l_depth_consist', type=float, default=1.0, help='lambda of depth consistency loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of Adam optimizer')
    parser.add_argument('--w_decay', type=float, default=0.0, help='weight decay of Adam optimizer')

    # Volumetric Primitive
    parser.add_argument('--sphere_num', type=int, default=16, help='number of spheres')
    parser.add_argument('--cuboid_num', type=int, default=0, help='number of cuboids')
    parser.add_argument('--vertex_num', type=int, default=128, help='number of vertices of each primitive')

    # Record Setting
    parser.add_argument('--output_path', type=str, default='')
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
from model.two_step import DepthEncoder, TranslateDecoder, VolumeRotateDecoder, DeformDecoder
from utils.sampling import Sampling
from utils.meshing import Meshing
from utils.loss import ChamferDistanceLoss
from utils.render import DepthRenderer
from utils.perceptual import get_local_features
from utils.visualize import save_depth_result, save_vp_result


def load_model(args):
    vp_num = args.cuboid_num + args.sphere_num
    depth_ae_path = 'checkpoint/depth_ae/depth_ae_epoch%03d.pth' % args.epoch \
        if not args.depth_ae_path else args.depth_ae_path
    depth_ae = DepthEstimationUNet().cuda()
    depth_ae.load_state_dict(torch.load(depth_ae_path))

    depth_en_path = 'checkpoint/depth_en/depth_en_epoch%03d.pth' % args.epoch \
        if not args.depth_en_path else args.depth_en_path
    depth_en = DepthEncoder().cuda()
    depth_en.load_state_dict(torch.load(depth_en_path))

    translate_de_path = 'checkpoint/translate_de/translate_de_epoch%03d.pth' % args.epoch \
        if not args.translate_de_path else args.translate_de_path
    translate_de = TranslateDecoder(vp_num=vp_num).cuda()
    translate_de.load_state_dict(torch.load(translate_de_path))

    local_feature_dim = 960

    volume_rotate_de_path = 'checkpoint/volume_rotate_de/volume_rotate_de_epoch%03d.pth' % args.epoch \
        if not args.volume_rotate_de_path else args.volume_rotate_de_path
    volume_rotate_de = VolumeRotateDecoder(feature_dim=local_feature_dim).cuda()
    volume_rotate_de.load_state_dict(torch.load(volume_rotate_de_path))

    return depth_ae, depth_en, translate_de, volume_rotate_de


def load_deform_de(args):
    local_feature_dim = 960
    global_feature_dim = 512

    deform_de_path = 'checkpoint/deform_de/deform_de_epoch%03d.pth' % args.epoch \
        if not args.deform_de_path else args.deform_de_path
    deform_de = DeformDecoder(feature_dim=global_feature_dim + local_feature_dim, vertex_num=args.vertex_num).cuda()
    deform_de.load_state_dict(torch.load(deform_de_path))

    return deform_de


def set_path(args):
    output_path = './output/eval/epoch%03d' % args.epoch if not args.output_path else args.output_path
    record_paths = {'loss': os.path.join(output_path, 'loss'),
                    'depth': os.path.join(output_path, 'depth'),
                    'vp': os.path.join(output_path, 'vp'),
                    'mesh': os.path.join(output_path, 'mesh'),
                    'ft': os.path.join(output_path, 'ft')}
    for record_path in list(record_paths.values()):
        os.makedirs(record_path, exist_ok=True)

    return record_paths


def eval(args):
    dataset = GenReDataset(args, 'test') if args.dataset == 'genre' else R2N2Dataset(args, 'test')
    print('Load %s testing dataset, size =' % args.dataset, len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=8, shuffle=False, collate_fn=collate_func)

    vp_num = args.cuboid_num + args.sphere_num
    record_paths = set_path(args)
    depth_ae, depth_en, translate_de, volume_rotate_de, = load_model(args)

    depth_ae.eval()
    depth_en.eval()
    translate_de.eval()
    volume_rotate_de.eval()

    mse_loss_func, cd_loss_func = MSELoss(), ChamferDistanceLoss()

    class_losses = {'depth': {}, 'vp_cd': {}, 'init_mesh_cd': {}, 'ft_mesh_cd': {}}
    class_n = {}

    progress_bar = tqdm(enumerate(dataloader))

    for idx, data in progress_bar:
        deform_de = load_deform_de(args)
        deform_de.eval()
        optimizer = Adam(params=deform_de.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        with torch.no_grad():
            rgbs, masks = data['rgb'].cuda(), data['mask'].cuda()  # (B, 3, H, W), (B, 1, H, W)
            class_ids = data['class_id']
            vertices = [one_vertices.cuda() for one_vertices in data['vertices']]  # list((N1, 3), ..., (Nb, 3))
            faces = [one_faces.cuda() for one_faces in data['faces']]
            # dists, elevs, azims = data['dist'].cuda(), data['elev'].cuda(), data['azim'].cuda()

            gt_depths = DepthRenderer.render_depths_of_multi_meshes(vertices, faces)
            gt_meshes = Meshing.meshing_vertices_faces(vertices, faces)
            gt_points = Sampling.sample_mesh_points(gt_meshes, sample_num=1024)

            rgbs = rgbs * masks
            pred_depths = depth_ae(rgbs)
            pred_depths = pred_depths * masks

            input_depths = gt_depths if args.use_gt_depth else pred_depths

            global_features, perceptual_feature_list = depth_en(input_depths)

            translates = translate_de(global_features)
            vp_center_points = torch.cat([t[:, None, :] for t in translates], 1)  # (B, K, 3)

            volumes, rotates = [], []
            vp_features = get_local_features(vp_center_points, input_depths, perceptual_feature_list)  # (B, K, F)

            for i in range(vp_num):
                one_vp_feature = vp_features[:, i, :]  # (B, F)
                volume, rotate = volume_rotate_de(one_vp_feature)

                volumes.append(volume)
                rotates.append(rotate)

            vp_meshes = Meshing.vp_meshing(volumes, rotates, translates,
                                             cuboid_num=args.cuboid_num, sphere_num=args.sphere_num)

            pred_coarse_points = Sampling.sample_mesh_points(vp_meshes, sample_num=1024)

            vp_cd_loss = cd_loss_func(pred_coarse_points, gt_points, each_batch=True)[0]

        for iter_now in range(len(args.iter_num)):
            deforms = []
            for i in range(vp_num):
                one_vp_feature = vp_features[:, i, :]  # (B, F)
                deform = deform_de(global_features, one_vp_feature)
                deforms.append(deform)

            pred_meshes = [TriangleMesh.from_tensors(m.vertices.clone(), m.faces.clone()) for m in vp_meshes]

            for i in range(vp_num):
                for b in range(len(pred_meshes)):
                    pred_meshes[b].vertices[i * args.vertex_num: (i + 1) * args.vertex_num, :] += deforms[i][b]

            if not iter_now:
                pred_fine_points = Sampling.sample_mesh_points(pred_meshes, sample_num=1024)
                init_mesh_cd_loss = cd_loss_func(pred_fine_points, gt_points, each_batch=True)[0]
                pred_init_meshes = [TriangleMesh.from_tensors(m.vertices.clone(), m.faces.clone()) for m in pred_meshes]

            if iter_now == args.iter_num - 1:
                pred_fine_points = Sampling.sample_mesh_points(pred_meshes, sample_num=1024)
                ft_mesh_cd_loss = cd_loss_func(pred_fine_points, gt_points, each_batch=True)[0]
            else:
                pred_vertices = [m.vertices for m in pred_meshes]
                pred_faces = [m.faces for m in pred_meshes]
                render_depths = DepthRenderer.render_depths_of_multi_meshes(pred_vertices, pred_faces)

                depth_consist_loss = mse_loss_func(render_depths, pred_depths) * args.l_depth_consist
                optimizer.zero_grad()
                depth_consist_loss.backward()
                optimizer.step()

        batch_size = rgbs.size(0)
        for b in range(batch_size):
            depth_loss = mse_loss_func(pred_depths[b][None], gt_depths[b][None])
            class_id = class_ids[b]

            if class_id in class_losses['depth']:
                class_losses['depth'][class_id] += depth_loss
                class_losses['vp_cd'][class_id] += vp_cd_loss[b]
                class_losses['init_mesh_cd'][class_id] += init_mesh_cd_loss[b]
                class_losses['ft_mesh_cd'][class_id] += ft_mesh_cd_loss[b]
                class_n[class_id] += 1
            else:
                class_losses['depth'][class_id] = depth_loss
                class_losses['vp_cd'][class_id] = vp_cd_loss[b]
                class_losses['init_mesh_cd'][class_id] = init_mesh_cd_loss[b]
                class_losses['ft_mesh_cd'][class_id] = ft_mesh_cd_loss[b]
                class_n[class_id] = 1

        if (idx + 1) % args.record_batch_interval == 0:
            depth_save_path = os.path.join(record_paths['depth'], 'batch%d.png' % (idx + 1))
            vp_save_path = os.path.join(record_paths['vp'], 'batch%d.png' % (idx + 1))
            mesh_save_path = os.path.join(record_paths['mesh'], 'batch%d.png' % (idx + 1))
            ft_save_path = os.path.join(record_paths['ft'], 'batch%d.png' % (idx + 1))

            save_depth_result(rgb=rgbs[0], mask=masks[0],
                              predict_depth=pred_depths[0], gt_depth=gt_depths[0], save_path=depth_save_path)
            save_vp_result(rgb=rgbs[0], mask=masks[0], input_depth=input_depths[0],
                           predict_mesh=vp_meshes[0], gt_mesh=gt_meshes[0],
                           vp_num=vp_num, vertex_num=args.vertex_num, save_path=vp_save_path)
            save_vp_result(rgb=rgbs[0], mask=masks[0], input_depth=input_depths[0],
                           predict_mesh=pred_init_meshes[0], gt_mesh=gt_meshes[0],
                           vp_num=vp_num, vertex_num=args.vertex_num, save_path=mesh_save_path)
            save_vp_result(rgb=rgbs[0], mask=masks[0], input_depth=input_depths[0],
                           predict_mesh=pred_meshes[0], gt_mesh=gt_meshes[0],
                           vp_num=vp_num, vertex_num=args.vertex_num, save_path=ft_save_path)

    avg_losses = {'depth': 0.0, 'vp_cd': 0.0, 'init_mesh_cd': 0.0, 'ft_mesh_cd': 0.0}
    headlines = {'depth': 'Depth MSE Loss', 'vp_cd': 'VP CD Loss',
                 'init_mesh_cd': 'Init Mesh CD Loss', 'ft_mesh_cd': 'FT Mesh CD Loss'}

    for key in list(avg_losses.keys()):
        print(headlines[key])
        print('id\t\tloss\t\tname')

        for obj_class in list(class_losses[key].keys()):
            class_losses[key][obj_class] = (class_losses[key][obj_class] / class_n[obj_class]).item()
            print('%s\t%.6f\t%s' % (obj_class, class_losses[key][obj_class], CLASS_DICT[obj_class]))
            avg_losses[key] += class_losses[key][obj_class]

        avg_losses[key] /= len(list(class_losses[key].keys()))
        print('total mean loss = %.6f' % avg_losses[key])
        class_losses[key]['total'] = avg_losses[key]

        np.savez(os.path.join(record_paths['loss'], '%s.npz' % key), **class_losses[key])


if __name__ == '__main__':
    eval(args)
