import torch
import numpy as np
import os
import glob
import pandas as pd
from plyfile import PlyData
from torch.utils.data import Dataset

class ArgoDataset(Dataset):

    def __init__(self, root, npoints, input_num, pred_num, seqs):
        super(ArgoDataset, self).__init__()

        self.root = root
        self.seqs = seqs
        self.input_num = input_num
        self.pred_num = pred_num
        self.npoints = npoints

        self.dataset = self.make_dataset()
    
    def get_cloud(self, filename):
        plydata = PlyData.read(filename)
        data = plydata.elements[0].data
        data_pd = pd.DataFrame(data)
        data_np = np.zeros(data_pd.shape, dtype=np.float32)
        property_names = data[0].dtype.names
        for i, name in enumerate(property_names):
            data_np[:,i] = data_pd[name]
        pc = data_np[:,:3]
        N = pc.shape[0]
        if N >= self.npoints:
            sample_idx = np.random.choice(N, self.npoints, replace=False)
        else:
            sample_idx = np.concatenate((np.arange(N), np.random.choice(N, self.npoints-N, replace=True)), axis=-1)
        pc = pc[sample_idx, :].astype('float32')
        pc = torch.from_numpy(pc).t()
        return pc
    
    def make_dataset(self):
        dataset = []
        
        for seq in self.seqs:
            dirs = os.listdir(os.path.join(self.root, seq))
            dirs = sorted(dirs)
            for curr_dir in dirs:
                names = os.listdir(os.path.join(self.root, seq, curr_dir, 'lidar'))
                names = sorted(names)
                max_ind = len(names)
                interval = self.input_num + self.pred_num
                ini_index = 0
                while (ini_index < max_ind - interval):
                    paths = []
                    
                    for j in range(interval):
                        curr_path = os.path.join(self.root, seq, curr_dir, 'lidar', names[j+ini_index])
                        paths.append(curr_path)
                        
                    ini_index += interval
                    dataset.append(paths)
                    
        return dataset
    
    def __getitem__(self, index):
        paths = self.dataset[index]

        input_pc_list = []
        
        for i in range(self.input_num):
            input_pc_path = paths[i]
            input_pc = self.get_cloud(input_pc_path)
            input_pc_list.append(input_pc)
            
        input_pc = torch.stack(input_pc_list, dim=0)

        output_pc_list = []
        
        for i in range(self.input_num, self.input_num+self.pred_num):
            output_pc_path = paths[i]
            output_pc = self.get_cloud(output_pc_path)
            output_pc_list.append(output_pc)
        output_pc = torch.stack(output_pc_list, dim=0)
        
        return input_pc, output_pc
        
    def __len__(self):
        return len(self.dataset)