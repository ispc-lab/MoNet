import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, grad
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from typing import Tuple, Union

import point_utils_cuda

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, knn_gather
import random
import emd_cuda

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        '''
        ctx:
        xyz: [B,N,3]
        npoint: int
        '''
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        point_utils_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output
    
    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthest_point_sample = FurthestPointSampling.apply

class WeightedFurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, weights: torch.Tensor, npoint: int) -> torch.Tensor:
        '''
        ctx:
        xyz: [B,N,3]
        weights: [B,N]
        npoint: int
        '''
        assert xyz.is_contiguous()
        assert weights.is_contiguous()
        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        point_utils_cuda.weighted_furthest_point_sampling_wrapper(B, N, npoint, xyz, weights, temp, output);
        return output
    
    @staticmethod
    def backward(xyz, a=None):
        return None, None

weighted_furthest_point_sample = WeightedFurthestPointSampling.apply

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        '''
        ctx
        features: [B,C,N]
        idx: [B,npoint]
        '''
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        point_utils_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()
        grad_features = Variable(torch.cuda.FloatTensor(B,C,N).zero_())
        grad_out_data = grad_out.data.contiguous()
        point_utils_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None

gather_operation = GatherOperation.apply

class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        point_utils_cuda.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

three_nn = ThreeNN.apply

class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        point_utils_cuda.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()

        point_utils_cuda.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None

three_interpolate = ThreeInterpolate.apply

def batch_chamfer_distance(pc1, pc2):
    '''
    Input:
        pc1: [B,3,N]
        pc2: [B,3,N]
    '''
    pc1 = pc1.permute(0,2,1).contiguous()
    pc2 = pc2.permute(0,2,1).contiguous()
    dist_batch, _ = chamfer_distance(pc1, pc2, batch_reduction='mean', point_reduction='mean')
    return dist_batch

def multi_frame_chamfer_loss(pc1, pc2_list):
    '''
    Calculate chamfer distance consecutive point cloud stream
    Input:
        pc1: [B,T,3,N]
        pc2_list: a list of [B,3,N]
    '''
    pred_num = len(pc2_list)
    l_total = 0
    for i in range(pred_num):
        curr_pc1 = pc1[:,i,:,:].squeeze(1).contiguous()
        curr_pc2 = pc2_list[i]

        curr_chamfer_dist = batch_chamfer_distance(curr_pc1, curr_pc2)
        l_total += curr_chamfer_dist
    l_chamfer = l_total/pred_num
    return l_chamfer

class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2

def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.

    Returns:
        cost (torch.Tensor): (b)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
    return cost

def EMD(pc1, pc2):
    '''
    Input:
        pc1: [1,3,M]
        pc2: [1,3,M]
    Ret:
        d: torch.float32
    '''
    pc1 = pc1.permute(0,2,1).contiguous()
    pc2 = pc2.permute(0,2,1).contiguous()
    d = earth_mover_distance(pc1, pc2, transpose=False)
    d = torch.mean(d)/pc1.shape[1]
    return d

def set_seed(seed):
    '''
    Set random seed for torch, numpy and python
    '''
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
        
    torch.backends.cudnn.benchmark=False 
    torch.backends.cudnn.deterministic=True