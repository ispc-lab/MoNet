import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from data.kitti_loader import KittiDataset
from data.argo_loader import ArgoDataset

from model.models import MoNet

from model.utils import batch_chamfer_distance, multi_frame_chamfer_loss, set_seed

from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MoNet')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
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
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')

    return parser.parse_args()

def validation(args, net):
    if args.dataset == 'kitti':
        val_seqs = ['06','07']
        val_dataset = KittiDataset(args.root, args.npoints, args.input_num, args.pred_num, val_seqs)
    elif args.dataset == 'argoverse':
        val_seqs = ['val']
        val_dataset = ArgoDataset(args.root, args.npoints, args.input_num, args.pred_num, val_seqs)
    else:
        raise('Not implemented')
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    net.eval()

    total_val_loss = 0
    count = 0
    pbar = tqdm(enumerate(val_loader))
    with torch.no_grad():
        for i, data in pbar:
            input_pc, output_pc = data
            input_pc = input_pc.cuda()
            output_pc = output_pc.cuda()

            pred_pc = net(input_pc)

            loss = multi_frame_chamfer_loss(output_pc[:,:,:3,:], pred_pc)
            total_val_loss += loss.item()
            count += 1
    
    total_val_loss = total_val_loss/count
    return total_val_loss

def train(args):
    if args.dataset == 'kitti':
        train_seqs = ['00','01','02','03','04','05']
        train_dataset = KittiDataset(args.root, args.npoints, args.input_num, args.pred_num, train_seqs)
    elif args.dataset == 'argoverse':
        train_seqs = ['train1', 'train2', 'train3', 'train4']
        train_dataset = ArgoDataset(args.root, args.npoints, args.input_num, args.pred_num, train_seqs)
    else:
        raise('Not implemented')
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    
    net = MoNet(args)

    if args.use_wandb:
        wandb.watch(net)
    if args.multi_gpu:
        net = torch.nn.DataParallel(net)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_train_epoch = 0
    best_val_epoch = 0

    for epoch in tqdm(range(args.epochs)):

        net.train()
        count = 0
        total_loss = 0
        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            input_pc, output_pc = data
            input_pc = input_pc.cuda()
            output_pc = output_pc.cuda()

            optimizer.zero_grad()
            pred_pc = net(input_pc)

            loss = multi_frame_chamfer_loss(output_pc[:,:,:3,:], pred_pc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=5.0)
            optimizer.step()

            count += 1
            total_loss += loss.item()

            if i % 10 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, i, len(train_loader), 100. * i/len(train_loader), loss.item()
                ))

        total_loss = total_loss/count
        total_val_loss = validation(args, net)

        if args.use_wandb:
            wandb.log({"train loss":total_loss, "val loss":total_val_loss})
        
        print('\n Epoch {} finished. Training loss: {:.4f} Valiadation loss: {:.4f}'.\
            format(epoch+1, total_loss, total_val_loss))

        ckpt_dir = os.path.join(args.ckpt_dir, 'ckpt_'+args.runname)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        if total_loss < best_train_loss:
            if args.multi_gpu:
                torch.save(net.module.state_dict(), os.path.join(ckpt_dir, 'best_train.pth'))
            else:
                torch.save(net.state_dict(), os.path.join(ckpt_dir, 'best_train.pth'))
            best_train_loss = total_loss
            best_train_epoch = epoch + 1
        
        if total_val_loss < best_val_loss:
            if args.multi_gpu:
                torch.save(net.module.state_dict(), os.path.join(ckpt_dir, 'best_val.pth'))
            else:
                torch.save(net.state_dict(), os.path.join(ckpt_dir, 'best_val.pth'))
            best_val_loss = total_val_loss
            best_val_epoch = epoch + 1
        
        print('Best train epoch: {} Best train loss: {:.4f} Best val epoch: {} Best val loss: {:.4f}'.format(
            best_train_epoch, best_train_loss, best_val_epoch, best_val_loss
        ))
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    if args.use_wandb:
        import wandb
        wandb.init(config=args, project='MoNet', name=args.dataset+'_'+args.runname, dir=args.wandb_dir)
    train(args)