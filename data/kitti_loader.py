import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset

class KittiDataset(Dataset):
    '''
    Multi sequence training on Kitti dataset
    Parameter:
        root: dir of kitti dataset (sequence/)
        npoints: number of random sampled points from raw points
        input_num: input point cloud number
        pred_num: predicted point cloud number
        seqs: sequence list
    '''
    def __init__(self, root, npoints, input_num, pred_num, seqs):
        super(KittiDataset, self).__init__()

        self.root = root
        self.seqs = seqs
        self.input_num = input_num
        self.pred_num = pred_num
        self.npoints = npoints
        self.dataset = self.make_dataset()
    
    def make_dataset(self):
        dataset = []
        for seq in self.seqs:
            dataroot = os.path.join(self.root, seq, 'velodyne')
            datapath = glob.glob(os.path.join(dataroot, '*.bin'))
            datapath = sorted(datapath)
            max_ind = len(datapath)
            ini_index = 0
            interval = self.input_num + self.pred_num
            while (ini_index < max_ind - interval):
                paths = []
                for i in range(interval):
                    curr_path = os.path.join(seq, 'velodyne',datapath[ini_index+i])
                    paths.append(curr_path)
                ini_index += interval
                dataset.append(paths)
        return dataset
    
    def get_cloud(self, filename):
        pc = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
        N = pc.shape[0]
        if N >= self.npoints:
            sample_idx = np.random.choice(N, self.npoints, replace=False)
        else:
            sample_idx = np.concatenate((np.arange(N), np.random.choice(N, self.npoints-N, replace=True)), axis=-1)
        pc = pc[sample_idx, :3].astype('float32')
        pc = torch.from_numpy(pc).t()
        return pc

    def __getitem__(self, index):
        paths = self.dataset[index]

        input_pc_list = []
        for i in range(self.input_num):
            input_pc_name = paths[i]
            input_pc = self.get_cloud(os.path.join(self.root, input_pc_name))
            input_pc_list.append(input_pc)
        input_pc = torch.stack(input_pc_list, dim=0)
        
        output_pc_list = []
        for i in range(self.input_num, self.input_num+self.pred_num):
            output_pc_name  = paths[i]
            output_pc = self.get_cloud(os.path.join(self.root, output_pc_name))
            output_pc_list.append(output_pc)
        output_pc = torch.stack(output_pc_list, dim=0)
        
        return input_pc, output_pc
    
    def __len__(self):
        return len(self.dataset)