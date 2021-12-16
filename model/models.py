import torch
import torch.nn as nn

from model.layers import MotionEncoder, ContentEncoder, MotionLSTM, MotionGRU, MotionAlign, PointNet2FeaturePropagator

class MoNet(nn.Module):
    '''
    Input:
        x: [B,T,3+C,N]
    '''
    def __init__(self, args):
        super(MoNet, self).__init__()

        self.rnn = args.rnn
        self.npoints = args.npoints
        self.pred_num = args.pred_num
        self.input_num = args.input_num

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256

        self.content_encoder_1 = ContentEncoder(npoints=int(self.npoints/32), k=32, in_channels=0, \
            out_channels=[int(self.C1/2),int(self.C1/2),self.C1], radius=0.5+1e-6, fps=True, knn=True)
        self.content_encoder_2 = ContentEncoder(npoints=int(self.npoints/64), k=16, in_channels=self.C1, \
            out_channels=[self.C1,self.C1,self.C2], radius=1.0+1e-6, fps=True, knn=True)
        self.content_encoder_3 = ContentEncoder(npoints=int(self.npoints/128), k=8, in_channels=self.C2, \
            out_channels=[self.C2,self.C2,self.C3], radius=2.0+1e-6, fps=True, knn=True)
        
        self.motion_encoder_1 = MotionEncoder(16, in_channels=self.C1, \
            out_channels=[self.C1,self.C1,self.C1])
        self.motion_encoder_2 = MotionEncoder(8, in_channels=self.C2, \
            out_channels=[self.C2,self.C2,self.C2])
        self.motion_encoder_3 = MotionEncoder(8, in_channels=self.C3, \
            out_channels=[self.C3,self.C3,self.C3])
        
        self.motion_align_1 = MotionAlign(16, self.C1)
        self.motion_align_2 = MotionAlign(16, self.C2)
        self.motion_align_3 = MotionAlign(16, self.C3)

        if self.rnn == 'LSTM':
            self.motion_rnn_1 = MotionLSTM(16, self.C1, self.C1, 2*self.C1)
            self.motion_rnn_2 = MotionLSTM(16, self.C2, self.C2, 2*self.C2)
            self.motion_rnn_3 = MotionLSTM(16, self.C3, self.C3, 2*self.C3)
        elif self.rnn == 'GRU':
            self.motion_rnn_1 = MotionGRU(16, self.C1, self.C1, 2*self.C1)
            self.motion_rnn_2 = MotionGRU(16, self.C2, self.C2, 2*self.C2)
            self.motion_rnn_3 = MotionGRU(16, self.C3, self.C3, 2*self.C3)
        else:
            raise('Not implemented')

        self.fp2 = PointNet2FeaturePropagator(2*self.C2, 2*self.C3, [2*self.C2], batchnorm=False)
        self.fp1 = PointNet2FeaturePropagator(2*self.C1, 2*self.C2, [2*self.C2], batchnorm=False)
        self.fp0 = PointNet2FeaturePropagator(0, 2*self.C2, [2*self.C2], batchnorm=False)
        
        self.classifier1 = nn.Conv1d(in_channels=2*self.C2, out_channels=128, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1, bias=False)
    
    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]

        # Embedding pipeline

        # Content encoder for input point clouds
        points_list_0 = []

        points_list_1 = []
        contents_list_1 = []
        points_list_2 = []
        contents_list_2 = []
        points_list_3 = []
        contents_list_3 = []

        for idx in range(self.input_num):
            points = x[:,idx,:,:].squeeze(1)
            points = points[:,:3,:].contiguous()
            points_list_0.append(points)

            points_1, contents_1 = self.content_encoder_1(points, None)
            points_2, contents_2 = self.content_encoder_2(points_1, contents_1)
            points_3, contents_3 = self.content_encoder_3(points_2, contents_2)

            points_list_1.append(points_1)
            contents_list_1.append(contents_1)
            points_list_2.append(points_2)
            contents_list_2.append(contents_2)
            points_list_3.append(points_3)
            contents_list_3.append(contents_3)
        
        # Motion encoder for input point clouds
        motion_list_1 = []
        motion_list_2 = []
        motion_list_3 = []
        
        for idx in range(self.input_num-1):
            motions_1 = self.motion_encoder_1(points_list_1[idx],contents_list_1[idx], \
                points_list_1[idx+1], contents_list_1[idx+1])
            motions_2 = self.motion_encoder_2(points_list_2[idx],contents_list_2[idx], \
                points_list_2[idx+1], contents_list_2[idx+1])
            motions_3 = self.motion_encoder_3(points_list_3[idx],contents_list_3[idx], \
                points_list_3[idx+1], contents_list_3[idx+1])
            
            motion_list_1.append(motions_1)
            motion_list_2.append(motions_2)
            motion_list_3.append(motions_3)

        # Initialize states for RNN
        if self.rnn == 'GRU':
            last_H1 = torch.zeros((B,2*self.C1,int(self.npoints/32)),dtype=torch.float32).cuda()
            last_H2 = torch.zeros((B,2*self.C2,int(self.npoints/64)),dtype=torch.float32).cuda()
            last_H3 = torch.zeros((B,2*self.C3,int(self.npoints/128)),dtype=torch.float32).cuda()
        elif self.rnn == 'LSTM':
            last_C1 = torch.zeros((B,2*self.C1,int(self.npoints/32)),dtype=torch.float32).cuda()
            last_C2 = torch.zeros((B,2*self.C2,int(self.npoints/64)),dtype=torch.float32).cuda()
            last_C3 = torch.zeros((B,2*self.C3,int(self.npoints/128)),dtype=torch.float32).cuda()
            last_H1 = torch.zeros((B,2*self.C1,int(self.npoints/32)),dtype=torch.float32).cuda()
            last_H2 = torch.zeros((B,2*self.C2,int(self.npoints/64)),dtype=torch.float32).cuda()
            last_H3 = torch.zeros((B,2*self.C3,int(self.npoints/128)),dtype=torch.float32).cuda()
        else:
            raise('Not implemented')

        curr_points_1 = torch.zeros_like(points_list_1[0], dtype=torch.float32).cuda()
        last_points_1 = torch.zeros_like(points_list_1[0], dtype=torch.float32).cuda()
        curr_points_2 = torch.zeros_like(points_list_2[0], dtype=torch.float32).cuda()
        last_points_2 = torch.zeros_like(points_list_2[0], dtype=torch.float32).cuda()
        curr_points_3 = torch.zeros_like(points_list_3[0], dtype=torch.float32).cuda()
        last_points_3 = torch.zeros_like(points_list_3[0], dtype=torch.float32).cuda()

        for idx in range(self.input_num-1):
            
            curr_motions_1 = motion_list_1[idx]
            curr_contents_1 = contents_list_1[idx]
            curr_motions_2 = motion_list_2[idx]
            curr_contents_2 = contents_list_2[idx]
            curr_motions_3 = motion_list_3[idx]
            curr_contents_3 = contents_list_3[idx]

            curr_points_1 = points_list_1[idx]
            curr_points_2 = points_list_2[idx]
            curr_points_3 = points_list_3[idx]

            if idx == 0:
                last_points_1 = torch.zeros_like(points_list_1[0], dtype=torch.float32).cuda()
                last_points_2 = torch.zeros_like(points_list_2[0], dtype=torch.float32).cuda()
                last_points_3 = torch.zeros_like(points_list_3[0], dtype=torch.float32).cuda()
            else:
                last_points_1 = points_list_1[idx-1]
                last_points_2 = points_list_2[idx-1]
                last_points_3 = points_list_3[idx-1]
            
            if self.rnn == 'LSTM':
                
                last_H1, last_C1 = self.motion_rnn_1(last_H1, last_C1, last_points_1, curr_points_1, curr_contents_1, curr_motions_1)
                last_H2, last_C2 = self.motion_rnn_2(last_H2, last_C2, last_points_2, curr_points_2, curr_contents_2, curr_motions_2)
                last_H3, last_C3 = self.motion_rnn_3(last_H3, last_C3, last_points_3, curr_points_3, curr_contents_3, curr_motions_3)
                
            
            elif self.rnn == 'GRU':
                last_H1 = self.motion_rnn_1(last_H1, last_points_1, curr_points_1, curr_contents_1, curr_motions_1)
                last_H2 = self.motion_rnn_2(last_H2, last_points_2, curr_points_2, curr_contents_2, curr_motions_2)
                last_H3 = self.motion_rnn_3(last_H3, last_points_3, curr_points_3, curr_contents_3, curr_motions_3)
            
            else:
                raise('Not implemented')
        
        # Inference pipeline

        # Initialization for inference
        last_points_1 = points_list_1[-2]
        last_points_2 = points_list_2[-2]
        last_points_3 = points_list_3[-2]

        curr_points_1 = points_list_1[-1]
        curr_points_2 = points_list_2[-1]
        curr_points_3 = points_list_3[-1]

        last_contents_1 = contents_list_1[-2]
        last_contents_2 = contents_list_2[-2]
        last_contents_3 = contents_list_3[-2]

        curr_contents_1 = contents_list_1[-1]
        curr_contents_2 = contents_list_2[-1]
        curr_contents_3 = contents_list_3[-1]

        curr_points_0 = points_list_0[-1]
        
        pred_points_list = []

        for idx in range(self.pred_num):

            curr_motions_1 = self.motion_align_1(last_points_1, curr_points_1, curr_motions_1)
            curr_motions_2 = self.motion_align_2(last_points_2, curr_points_2, curr_motions_2)
            curr_motions_3 = self.motion_align_3(last_points_3, curr_points_3, curr_motions_3)

            if self.rnn == 'LSTM':
                last_H1, last_C1 = self.motion_rnn_1(last_H1, last_C1, last_points_1, curr_points_1, curr_contents_1, curr_motions_1)
                last_H2, last_C2 = self.motion_rnn_2(last_H2, last_C2, last_points_2, curr_points_2, curr_contents_2, curr_motions_2)
                last_H3, last_C3 = self.motion_rnn_3(last_H3, last_C3, last_points_3, curr_points_3, curr_contents_3, curr_motions_3)
            
            elif self.rnn == 'GRU':
                last_H1 = self.motion_rnn_1(last_H1, last_points_1, curr_points_1, curr_contents_1, curr_motions_1)
                last_H2 = self.motion_rnn_2(last_H2, last_points_2, curr_points_2, curr_contents_2, curr_motions_2)
                last_H3 = self.motion_rnn_3(last_H3, last_points_3, curr_points_3, curr_contents_3, curr_motions_3)
            
            else:
                raise('Not implemented')

            # decoder
            l2_feat = self.fp2(curr_points_2.permute(0,2,1).contiguous(), curr_points_3.permute(0,2,1).contiguous(), last_H2, last_H3)
            l1_feat = self.fp1(curr_points_1.permute(0,2,1).contiguous(), curr_points_2.permute(0,2,1).contiguous(), last_H1, l2_feat)
            l0_feat = self.fp0(curr_points_0.permute(0,2,1).contiguous(), curr_points_1.permute(0,2,1).contiguous(), None, l1_feat)

            pred_flow = self.classifier2(self.classifier1(l0_feat))

            pred_points = curr_points_0 + pred_flow
            pred_points_list.append(pred_points)

            curr_points_0 = pred_points

            last_points_1 = curr_points_1
            last_points_2 = curr_points_2
            last_points_3 = curr_points_3
            last_contents_1 = curr_contents_1
            last_contents_2 = curr_contents_2
            last_contents_3 = curr_contents_3

            curr_points_1, curr_contents_1 = self.content_encoder_1(curr_points_0, None)
            curr_points_2, curr_contents_2 = self.content_encoder_2(curr_points_1, curr_contents_1)
            curr_points_3, curr_contents_3 = self.content_encoder_3(curr_points_2, curr_contents_2)

            curr_motions_1 = self.motion_encoder_1(last_points_1, last_contents_1, curr_points_1, curr_contents_1)
            curr_motions_2 = self.motion_encoder_2(last_points_2, last_contents_2, curr_points_2, curr_contents_2)
            curr_motions_3 = self.motion_encoder_3(last_points_3, last_contents_3, curr_points_3, curr_contents_3)

        return pred_points_list