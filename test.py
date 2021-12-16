import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.kitti_loader import KittiDataset
from data.argo_loader import ArgoDataset

from model.models import MoNet

from model.utils import batch_chamfer_distance, multi_frame_chamfer_loss, set_seed, EMD

from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='MoNet')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--runname', type=str, default='')
    parser.add_argument('--rnn', type=str, default='', help='LSTM/GRU')
    parser.add_argument('--pred_num', type=int, default=5)
    parser.add_argument('--input_num', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--ckpt', type=str, default='')

    return parser.parse_args()

def test(args):
    if args.dataset == 'kitti':
        test_seqs = ['08','09','10']
        test_dataset = KittiDataset(args.root, args.npoints, args.input_num, args.pred_num, test_seqs)
    elif args.dataset == 'argoverse':
        test_seqs = ['test']
        test_dataset = ArgoDataset(args.root, args.npoints, args.input_num, args.pred_num, test_seqs)
    else:
        raise('Not implemented')
    
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)
    
    net = MoNet(args)
    net.cuda()
    net.load_state_dict(torch.load(args.ckpt))

    count = 0
    l_chamfer_list = [0.0] * args.pred_num
    l_emd_list = [0.0] * args.pred_num

    pbar = tqdm(enumerate(test_loader))

    net.eval()

    with torch.no_grad():
        for i, data in pbar:
            input_pc, output_pc = data
            input_pc = input_pc.cuda()
            output_pc = output_pc.cuda()

            pred_pc = net(input_pc)

            for t in range(args.pred_num):
                l_chamfer_one = batch_chamfer_distance(output_pc[:,t,:3,:].squeeze(1), pred_pc[t])
                l_emd_one = EMD(output_pc[:,t,:3,:].squeeze(1), pred_pc[t])
                l_chamfer_list[t] += l_chamfer_one.item()
                l_emd_list[t] += l_emd_one.item()
                l_emd_list[t] += 0.0

            count += 1

        for t in range(args.pred_num):
            l_chamfer_list[t] = l_chamfer_list[t]/count
            l_emd_list[t] = l_emd_list[t]/count
    
    print('Chamfer Distance:', l_chamfer_list)
    print('Average Chamfer Distance:', np.mean(np.array(l_chamfer_list)))
    print('Earth Mover Distance', l_emd_list)
    print('Average Earth Mover Distance', np.mean(np.array(l_emd_list)))

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    test(args)