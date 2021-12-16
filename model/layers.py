import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.utils import furthest_point_sample, gather_operation, three_nn, three_interpolate
from pytorch3d.ops import knn_points, knn_gather

def knn_group(points1, points2, k):
    '''
    For each point in points1, query k-nearest-neighbors from points2,

    Input:
        points1: [B,3,M] (anchor points)
        points2: [B,3,N] (query points)
    Output:
        nn_group: [B,4,M,k]
        nn_idx: [B,M,k]
    '''
    points1 = points1.permute(0,2,1).contiguous() # [B,M,3]
    points2 = points2.permute(0,2,1).contiguous() # [B,N,3]
    _, nn_idx, nn_points = knn_points(points1, points2, K=k, return_nn=True)
    points1_expand = points1.unsqueeze(2).repeat(1,1,k,1)
    rela_nn = nn_points - points1_expand # [B,M,k,3]
    rela_dist = torch.norm(rela_nn, dim=-1, keepdim=True) # [B,M,k,1]
    nn_group = torch.cat((rela_nn, rela_dist), dim=-1) # [B,M,k,4]
    nn_group = nn_group.permute(0,3,1,2).contiguous()
    return nn_group, nn_idx

class MotionAlign(nn.Module):
    '''
    Input:
        points1: [B,3,N]
        points2: [B,3,N]
        motion1: [B,C,N]
    Output:
        aligned_motion: [B,C,N]
    '''
    def __init__(self, k, feature_size):
        super(MotionAlign, self).__init__()
        self.k = k
        self.feature_size = feature_size
        self.mlps = nn.Sequential(nn.Conv2d(self.feature_size+4, int(self.feature_size/2), kernel_size=1, bias=True),
                                  nn.Conv2d(int(self.feature_size/2), int(self.feature_size/4), kernel_size=1, bias=True))
    
    def forward(self, points1, points2, motion1):
        nn_group, nn_idx = knn_group(points2, points1, self.k)
        features2 = knn_gather(motion1.permute(0,2,1), nn_idx).permute(0,3,1,2).contiguous()
        weights = self.mlps(torch.cat([nn_group,features2],dim=1))
        weights = torch.max(weights, dim=1, keepdim=True)[0]
        weights = F.softmax(weights, dim=-1) # [B,1,N,k]
        aligned_motion = torch.sum(torch.mul(features2, weights.repeat(1,self.feature_size,1,1)),dim=-1)
        return aligned_motion

class MotionGRU(nn.Module):
    '''
    Parameters:
        k: k nearest neighbors
        content_size: content feature size of current frame
        motion_size: motion feature size of current frame
        hidden_size: output feature size of hidden state
    Input:
        H0: [B,C,N] (hidden state of last frame)
        points0: [B,3,N] (point coordinates of last frame)
        points1: [B,3,N] (point coordinates of current frame)
        contents1: [B,C1,N] (content features of current frame)
        motions1: [B,C2,N] (motion features of current frame)

    Output:
        H1: hidden state
    '''
    def __init__(self, k, content_size, motion_size, hidden_size):
        super(MotionGRU, self).__init__()
        self.k = k
        self.feature_size = content_size + motion_size
        self.hidden_size = hidden_size

        self.mlp_R = nn.Sequential(nn.Conv2d(self.hidden_size+self.feature_size+4, self.hidden_size, kernel_size=1, bias=True))
        self.mlp_Z = nn.Sequential(nn.Conv2d(self.hidden_size+self.feature_size+4, self.hidden_size, kernel_size=1, bias=True))
        
        self.mlp_H1_0 = nn.Sequential(nn.Conv2d(self.hidden_size+4, self.hidden_size, kernel_size=1, bias=True))
        self.mlp_H1_1 = nn.Sequential(nn.Conv1d(self.hidden_size+self.feature_size, self.hidden_size, kernel_size=1, bias=True))
    
    def forward(self, H0, points0, points1, contents1, motions1):

        features1 = torch.cat([contents1, motions1], dim=1)

        nn_group, nn_idx = knn_group(points1, points0, self.k) # [B,4+C,N,k]
        nn_H0 = knn_gather(H0.permute(0,2,1), nn_idx).permute(0,3,1,2).contiguous() 
        features1_expand = features1.unsqueeze(-1).repeat(1,1,1,self.k)

        gate_R = self.mlp_R(torch.cat((nn_group,nn_H0,features1_expand),dim=1)) # [B,C,N,k]
        gate_R = torch.sigmoid(torch.max(gate_R, dim=-1, keepdim=False)[0]) # [B,C,N]

        gate_Z = self.mlp_Z(torch.cat((nn_group,nn_H0,features1_expand),dim=1)) # [B,C,N,k]
        gate_Z = torch.sigmoid(torch.max(gate_Z, dim=-1, keepdim=False)[0]) # [B,C,N]

        H1_0 = self.mlp_H1_0(torch.cat((nn_group,nn_H0), dim=1)) # [B,C,N,k]
        H1_0 = torch.max(H1_0, dim=-1, keepdim=False)[0] # [B,C,N]

        H1_1 = torch.tanh(self.mlp_H1_1(torch.cat((features1,torch.mul(gate_R,H1_0)),dim=1)))
        H1 = torch.mul(gate_Z,H1_0) + torch.mul(1.0-gate_Z,H1_1)

        return H1

class MotionLSTM(nn.Module):
    '''
    Parameters:
        k: k nearest neighbors
        content_size: content feature size of current frame
        motion_size: motion feature size of current frame
        hidden_size: output feature size of hidden state
    Input:
        H0: [B,C,N] (hidden state of last frame)
        C0: [B,C,N] (cell state of last frame)
        points0: [B,3,N] (point coordinates of last frame)
        points1: [B,3,N] (point coordinates of current frame)
        contents1: [B,C1,N] (content features of current frame)
        motions1: [B,C2,N] (motion features of current frame)

    Output:
        H1: hidden state
        C1: cell state
    '''
    def __init__(self, k, content_size, motion_size, hidden_size):
        super(MotionLSTM, self).__init__()
        self.k = k
        self.feature_size = content_size + motion_size
        self.hidden_size = hidden_size

        self.mlp_I = nn.Sequential(nn.Conv2d(self.hidden_size+self.feature_size+4, self.hidden_size, kernel_size=1, bias=True))
        self.mlp_F = nn.Sequential(nn.Conv2d(self.hidden_size+self.feature_size+4, self.hidden_size, kernel_size=1, bias=True))
        self.mlp_O = nn.Sequential(nn.Conv2d(self.hidden_size+self.feature_size+4, self.hidden_size, kernel_size=1, bias=True))
        
        self.mlp_C0 = nn.Sequential(nn.Conv2d(self.hidden_size+4, self.hidden_size, kernel_size=1, bias=True))

        self.mlp_C1_1 = nn.Sequential(nn.Conv2d(self.hidden_size+self.feature_size+4, self.hidden_size, kernel_size=1, bias=True))
    
    def forward(self, H0, C0, points0, points1, contents1, motions1):

        features1 = torch.cat([contents1, motions1], dim=1)

        nn_group, nn_idx = knn_group(points1, points0, self.k) # [B,4+C,N,k]
        nn_H0 = knn_gather(H0.permute(0,2,1), nn_idx).permute(0,3,1,2).contiguous() # [B,C,N,k]
        nn_C0 = knn_gather(C0.permute(0,2,1), nn_idx).permute(0,3,1,2).contiguous() # [B,C,N,k]

        features1 = features1.unsqueeze(-1).repeat(1,1,1,self.k)

        gate_I = self.mlp_I(torch.cat((nn_group,nn_H0,features1), dim=1)) # [B,C,N,k]
        gate_I = torch.sigmoid(torch.max(gate_I, dim=-1, keepdim=False)[0]) # [B,C,N]

        gate_F = self.mlp_F(torch.cat((nn_group,nn_H0,features1), dim=1))
        gate_F = torch.sigmoid(torch.max(gate_F, dim=-1, keepdim=False)[0]) # [B,C,N]

        gate_O = self.mlp_O(torch.cat((nn_group,nn_H0,features1), dim=1))
        gate_O = torch.sigmoid(torch.max(gate_O, dim=-1, keepdim=False)[0]) # [B,C,N]

        C1_0 = self.mlp_C0(torch.cat((nn_group, nn_C0), dim=1))
        C1_0 = torch.max(C1_0, dim=-1, keepdim=False)[0] # [B,C,N]

        C1_1 = self.mlp_C1_1(torch.cat((nn_group,nn_H0,features1), dim=1))
        C1_1 = torch.tanh(torch.max(C1_1, dim=-1, keepdim=False)[0]) # [B,C,N]

        C1 = torch.mul(gate_F, C1_0) + torch.mul(gate_I, C1_1) # [B,C,N]
        H1 = torch.mul(gate_O, torch.tanh(C1)) # [B,C,N]

        return H1, C1

class FurthestPointsSample(nn.Module):
    '''
    Furthest point sampling
    Parameters:
        npoints: number of sampled points
    Input:
        x: [B,3,N]
    Output:
        fps_points: [B,3,npoints]
    '''
    def __init__(self, npoints):
        super(FurthestPointsSample, self).__init__()
        self.npoints = npoints
    
    def forward(self, x):
        fps_points_ind = furthest_point_sample(x.permute(0,2,1).contiguous(), self.npoints)
        fps_points = gather_operation(x, fps_points_ind)

        return fps_points

class ContentEncoder(nn.Module):
    '''
    Parameters:
        npoints: number of sample points
        k: k nearest number
        in_channels: input feature channels (C_in)
        out_channels: output feature channels (C_out)
        fps: True/False
        knn: True/False
    Input:
        points: [B,3,N]
        features: [B,C_in,N]
    Output:
        fps_points: [B,3,npoints]
        output_features: [B,C_out,npoints]
    '''
    def __init__(self, npoints, k, in_channels, out_channels, radius, fps=True, knn=True):
        super(ContentEncoder, self).__init__()
        self.k = k
        self.fps = fps
        self.knn = knn
        self.furthest_points_sample = FurthestPointsSample(npoints)
        self.radius = radius

        layers = []
        out_channels = [in_channels+4,*out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=True),
                       nn.ReLU()]
        self.conv = nn.Sequential(*layers)
    
    def forward(self, points, features):
        
        fps_points = self.furthest_points_sample(points) # [B,3,npoints]
        if self.knn:
            nn_group, nn_idx = knn_group(fps_points, points, self.k) # [B,4,npoints,k]
            if features is not None:
                nn_features = knn_gather(features.permute(0,2,1), nn_idx).permute(0,3,1,2).contiguous() # [B,C_in,npoints,k]
        else:
            raise NotImplementedError
        
        if features is not None:
            new_features = torch.cat([nn_group, nn_features], dim=1) # [B,C_in+4,npoints,k]
        else:
            new_features = nn_group
        new_features = self.conv(new_features) # [B,C_out,npoints,k]
        out_features = torch.max(new_features, dim=-1, keepdim=False)[0] #  [B,C_out,npoints]

        return fps_points, out_features

class MotionEncoder(nn.Module):
    '''
    Parameters:
        k: k nearest neighbors
        in_channels: input feature channels
        out_channels: output feature channels
    Input:
        points1: [B,3,N]
        features1: [B,C,N]
        points2: [B,3,N]
        features2: [B,C,N]
    Output:
        motions: [B,C_out,N]
    '''
    def __init__(self, k, in_channels, out_channels):
        super(MotionEncoder, self).__init__()

        self.k = k

        layers = []

        out_channels = [2*in_channels+4, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=True),
                       nn.ReLU()]
        self.conv = nn.Sequential(*layers)
    
    def forward(self, points1, features1, points2, features2):
        
        nn_group, nn_idx = knn_group(points1, points2, self.k)
        nn_features2 = knn_gather(features2.permute(0,2,1), nn_idx).permute(0,3,1,2).contiguous()
        new_features = torch.cat([nn_group, nn_features2, features1.unsqueeze(3).repeat(1,1,1,self.k)],dim=1) # [B,4+C+C,N,k]
        new_features = self.conv(new_features) # [B,C_out,N,k]
        motions = torch.max(new_features,dim=-1)[0]

        return motions

class PointNet2FeaturePropagator(nn.Module):
    '''
    Parameters:
        in_channels1: input feature channels 1
        in_channels2: input feature channels 2
        out_channels: output feature channels
    Input:
        xyz: [B,N,3]
        xyz_prev: [B,N,3]
        features: [B,C,N]
        features_prev: [B,C,N]
    '''

    def __init__(self, in_channels1, in_channels2, out_channels, batchnorm=True):
        super(PointNet2FeaturePropagator, self).__init__()

        self.layer_dims = out_channels

        unit_pointnets = []
        in_channels = in_channels1 + in_channels2
        for out_channel in out_channels:
            unit_pointnets.append(
                nn.Conv1d(in_channels, out_channel, 1))

            if batchnorm:
                unit_pointnets.append(nn.BatchNorm1d(out_channel))

            unit_pointnets.append(nn.ReLU())
            in_channel = out_channel

        self.unit_pointnet = nn.Sequential(*unit_pointnets)

    def forward(self, xyz, xyz_prev, features=None, features_prev=None):
        """
        Args:
            xyz (torch.Tensor): shape = (batch_size, num_points, 3)
                The 3D coordinates of each point at current layer,
                computed during feature extraction (i.e. set abstraction).
            xyz_prev (torch.Tensor|None): shape = (batch_size, num_points_prev, 3)
                The 3D coordinates of each point from the previous feature
                propagation layer (corresponding to the next layer during
                feature extraction).
                This value can be None (i.e. for the very first propagator layer).
            features (torch.Tensor|None): shape = (batch_size, num_features, num_points)
                The features of each point at current layer,
                computed during feature extraction (i.e. set abstraction).
            features_prev (torch.Tensor|None): shape = (batch_size, num_features_prev, num_points_prev)
                The features of each point from the previous feature
                propagation layer (corresponding to the next layer during
                feature extraction).
        Returns:
            (torch.Tensor): shape = (batch_size, num_features_out, num_points)
        """
        num_points = xyz.shape[1]
        if xyz_prev is None:  # Very first feature propagation layer
            new_features = features_prev.expand(
                *(features.shape + [num_points]))

        else:
            dist, idx = three_nn(xyz, xyz_prev)
            # shape = (batch_size, num_points, 3), (batch_size, num_points, 3)
            inverse_dist = 1.0 / (dist + 1e-8)
            total_inverse_dist = torch.sum(inverse_dist, dim=2, keepdim=True)
            weights = inverse_dist / total_inverse_dist
            new_features = three_interpolate(features_prev, idx, weights)
            # shape = (batch_size, num_features_prev, num_points)

        if features is not None:
            new_features = torch.cat([new_features, features], dim=1)

        return self.unit_pointnet(new_features)