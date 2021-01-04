### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy
from .dconv.modules.modulated_deform_conv import ModulatedDeformConvPack
import time
import random

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):        
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_warper(input_nc_1, input_nc_2, input_nc_3, ngf, n_downsampling, norm, scale, gpu_ids=[], opt=[]):
    net = None    
    norm_layer = get_norm_layer(norm_type=norm)

    net = ClothWarper(opt, input_nc_1, input_nc_2, input_nc_3, ngf, n_downsampling, opt.n_blocks, opt.fg, opt.no_flow, norm_layer)

    #print_network(netG)
    if len(gpu_ids) > 0:
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net

def define_composer(input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, norm, scale, gpu_ids=[], opt=[]):
    net = None    
    norm_layer = get_norm_layer(norm_type=norm)

    net = Composer(opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, opt.n_blocks, opt.fg, opt.no_flow, norm_layer)

    #print_network(netG)
    if len(gpu_ids) > 0:
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net

def define_parser(input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, norm, scale, gpu_ids=[], opt=[]):
    net = None    
    norm_layer = get_norm_layer(norm_type=norm)

    net = Parser(opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, opt.n_blocks, opt.fg, opt.no_flow, norm_layer)

    #print_network(netG)
    if len(gpu_ids) > 0:
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net

def define_D(input_nc, ndf, n_layers_D, norm='instance', num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, num_D, getIntermFeat)   
    #print_network(netD)
    if len(gpu_ids) > 0:    
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    if dtype == torch.float16: t_grid = t_grid.half()
    return t_grid.cuda(gpu_id)

##############################################################################
# Classes
##############################################################################
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def grid_sample(self, input1, input2):
        if self.opt.fp16: # not sure if it's necessary
            return torch.nn.functional.grid_sample(input1.float(), input2.float(), mode='bilinear', padding_mode='border').half()
        else:
            return torch.nn.functional.grid_sample(input1, input2, mode='bilinear', padding_mode='border')

    def resample(self, image, flow, normalize=True):        
        b, c, h, w = image.size()        
        if not hasattr(self, 'grid') or self.grid.size() != flow.size():
            self.grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
        if normalize:
            flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
        final_grid = (self.grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
        output = self.grid_sample(image, final_grid)
        return output

##############################################################################
# Classes for coarse TPS warping
##############################################################################
class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
    
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B, LO_A, LO_B):
        b,c,h,w = feature_A.size()
        n_class = LO_A.size()[1]
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        #b,haxwa,hb,wb(b,wa,ha,hb,wb)
        LO_A = LO_A.transpose(2,3).contiguous().view(b,n_class,h*w)
        LO_B = LO_B.view(b,n_class,h*w).transpose(1,2)
        LO_mul = torch.bmm(LO_B, LO_A)

        feature_mul = feature_mul * LO_mul
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)

        return correlation_tensor

class FeatureCorrelation_wolo(nn.Module):
    def __init__(self):
        super(FeatureCorrelation_wolo, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor
    
class FeatureRegression(nn.Module):
    def __init__(self, input_nc_1=192,input_nc_2=512,output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc_1, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc_2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.conv_later = nn.Sequential(nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)
        self.linear = nn.Linear(64 * 4 * 3, output_dim)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x, x_1):

        x = self.conv(x) + self.conv1(x_1)
        x = self.conv_later(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x
        
class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        '''
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()
        '''
        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            # P_X, P_Y: (N,1)
            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0)
            # P_X, P_Y: (1,1,1,1,N)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            '''
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()
            '''
            
    def forward(self, theta):
        gpu_id = theta.get_device()
        original_grid = torch.cat((self.grid_X.cuda(gpu_id),self.grid_Y.cuda(gpu_id)),3)
        warped_grid = self.apply_transformation(theta,original_grid)
        
        return original_grid, warped_grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        #K: (N,N)
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        #ones
        O = torch.FloatTensor(N,1).fill_(1)
        #zeros
        Z = torch.FloatTensor(3,3).fill_(0)  
        # (N,3)     
        P = torch.cat((O,X,Y),1)
        # (N,N+3), (3,N+3) -> (N+3,N+3)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        #if self.use_cuda:
        #    Li = Li.cuda()
        return Li
        
    def apply_transformation(self,theta,points):
        gpu_id = theta.get_device()
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        Q_X = Q_X + self.P_X_base.cuda(gpu_id).expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.cuda(gpu_id).expand_as(Q_Y)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.cuda(gpu_id).expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.cuda(gpu_id).expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].cuda(gpu_id).expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].cuda(gpu_id).expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].cuda(gpu_id).expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].cuda(gpu_id).expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)

##############################################################################
# Class for our C2F-FWN
##############################################################################
class ClothWarper(BaseNetwork):
    #input_nc_1(1X): TPose, TLO, TFG; input_nc_2(1X): SPose, SLO; input_nc_3(2X):（SLO + SP + (X)shadow map +) total flow (TODO: 2X can change to 1X to enable dconv)
    def __init__(self, opt, input_nc_1, input_nc_2, input_nc_3, ngf, n_downsampling=4, n_blocks=9, use_fg_model=False, no_flow=False,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(ClothWarper, self).__init__()
        self.opt = opt
        self.n_downsampling = n_downsampling
        activation = nn.ReLU(True)

        model_down_target_0 = [nn.Conv2d(input_nc_1, ngf, kernel_size=4, stride=2, padding=1), activation, norm_layer(ngf)]
        self.model_down_target_0 = nn.Sequential(*model_down_target_0)
        model_down_source_0 = [nn.Conv2d(input_nc_2, ngf, kernel_size=4, stride=2, padding=1), activation, norm_layer(ngf)]
        self.model_down_source_0 = nn.Sequential(*model_down_source_0)
        model_down_prev_0 = [nn.Conv2d(input_nc_3, ngf, kernel_size=4, stride=2, padding=1), activation, norm_layer(ngf)]
        self.model_down_prev_0 = nn.Sequential(*model_down_prev_0)
        for i in range(self.n_downsampling-1):
            in_ngf = 2**i * ngf
            out_ngf = 2**(i+1) * ngf
            downconv_deform = ModulatedDeformConvPack(in_ngf, out_ngf, kernel_size=(4,4), stride=2, padding=1, deformable_groups=1, bias=True)
            model_down_target = [downconv_deform, activation]
            model_down_target += [norm_layer(out_ngf)]
            model_resdown_target = [ResnetBlock_deform(out_ngf, padding_type='zero', activation=activation, norm_layer=norm_layer)]
            setattr(self, 'model_down_target_'+str(i+1), nn.Sequential(*model_down_target))
            setattr(self, 'model_resdown_target_'+str(i+1), nn.Sequential(*model_resdown_target))
            model_down_source = copy.deepcopy(model_down_target)
            model_resdown_source = copy.deepcopy(model_resdown_target)
            setattr(self, 'model_down_source_'+str(i+1), nn.Sequential(*model_down_source))
            setattr(self, 'model_resdown_source_'+str(i+1), nn.Sequential(*model_resdown_source))
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model_down_prev = [downconv, activation]
            model_down_prev += [norm_layer(out_ngf)]
            model_down_prev += [ResnetBlock(out_ngf, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            setattr(self, 'model_down_prev_'+str(i+1), nn.Sequential(*model_down_prev))
        
        model_feat_target = [nn.Conv2d(8*ngf, 8*ngf, kernel_size=3, stride=1, padding=1), activation]
        model_feat_target += [norm_layer(8*ngf)]
        model_feat_target += [nn.Conv2d(8*ngf, 8*ngf, kernel_size=3, stride=1, padding=1), activation]
        model_feat_source = copy.deepcopy(model_feat_target)
        model_feat_prev = copy.deepcopy(model_feat_target)
        self.model_feat_target = nn.Sequential(*model_feat_target)
        self.model_feat_source = nn.Sequential(*model_feat_source)
        self.model_feat_prev = nn.Sequential(*model_feat_prev)

        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation() #(b, h*w, h, w)
        self.regression = FeatureRegression(input_nc_1=192, input_nc_2=512, output_dim=2*opt.grid_size**2, use_cuda=True)
        self.gridGen = TpsGridGen(opt.fine_height, opt.fine_width, use_cuda=True, grid_size=opt.grid_size)

        self.feat_3_T = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.feat_2_T = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.feat_1_T = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.feat_0_T = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.smooth_2_T = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth_1_T = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth_0_T = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.feat_3_S = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.feat_2_S = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.feat_1_S = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.feat_0_S = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.smooth_2_S = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth_1_S = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth_0_S = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.feat_3_P = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.feat_2_P = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.feat_1_P = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.feat_0_P = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.smooth_2_P = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth_1_P = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth_0_P = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        model_dense_flow = [nn.ConvTranspose2d(64*3, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm_layer(64), activation]
        model_dense_flow += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]

        self.model_dense_flow = nn.Sequential(*model_dense_flow)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, input_target, input_source, input_prev, input_tfg, input_grid=None):
        bsize, _, height, width = input_target.size()
        input_tlo = input_target[:, -1-self.opt.label_nc_2:-1]
        input_slo = input_source[:, -self.opt.label_nc_2:]

        input_tlo_0 = F.interpolate(input_tlo, scale_factor=0.5)
        input_tlo_1 = F.interpolate(input_tlo_0, scale_factor=0.5)
        input_tlo_2 = F.interpolate(input_tlo_1, scale_factor=0.5)
        input_tlo_3 = F.interpolate(input_tlo_2, scale_factor=0.5)

        input_slo_0 = F.interpolate(input_slo, scale_factor=0.5)
        input_slo_1 = F.interpolate(input_slo_0, scale_factor=0.5)
        input_slo_2 = F.interpolate(input_slo_1, scale_factor=0.5)
        input_slo_3 = F.interpolate(input_slo_2, scale_factor=0.5)

        gpu_id = input_target.get_device()
        feature_prev_0 = self.model_down_prev_0(input_prev)
        feature_prev_1 = self.model_down_prev_1(feature_prev_0)
        feature_prev_2 = self.model_down_prev_2(feature_prev_1)
        feature_prev_3 = self.model_down_prev_3(feature_prev_2)
        feature_prev = self.model_feat_prev(feature_prev_3)

        feature_T_0 = self.model_down_target_0(input_target)
        feature_T_1 = self.model_down_target_1([feature_T_0, input_tlo_0, input_tlo_1])
        feature_T_1 = self.model_resdown_target_1([feature_T_1, input_tlo_1])
        feature_T_2 = self.model_down_target_2([feature_T_1, input_tlo_1, input_tlo_2])
        feature_T_2 = self.model_resdown_target_2([feature_T_2, input_tlo_2])
        feature_T_3 = self.model_down_target_3([feature_T_2, input_tlo_2, input_tlo_3])
        feature_T_3 = self.model_resdown_target_3([feature_T_3, input_tlo_3])
        feature_T = self.model_feat_target(feature_T_3)

        feature_S_0 = self.model_down_source_0(input_source)
        feature_S_1 = self.model_down_source_1([feature_S_0, input_slo_0, input_slo_1])
        feature_S_1 = self.model_resdown_source_1([feature_S_1, input_slo_1])
        feature_S_2 = self.model_down_source_2([feature_S_1, input_slo_1, input_slo_2])
        feature_S_2 = self.model_resdown_source_2([feature_S_2, input_slo_2])
        feature_S_3 = self.model_down_source_3([feature_S_2, input_slo_2, input_slo_3])
        feature_S_3 = self.model_resdown_source_3([feature_S_3, input_slo_3])
        feature_S = self.model_feat_source(feature_S_3)

        feature_T = self.l2norm(feature_T)
        feature_S = self.l2norm(feature_S)
        correlation = self.correlation(feature_S, feature_T, input_slo_3, input_tlo_3)
        theta = self.regression(correlation, feature_prev)
        #need an original grid to compute tps flow
        original_grid, warped_grid = self.gridGen(theta)
        ###flow_tps is normalized to -1,1, while the estimated dense flow is not -> flow_tps need to be fixed
        flow_tps = warped_grid - original_grid
        flow_tps = flow_tps.permute(0,3,1,2)

        #downsample tps flow
        #since flow has already been normalized to -1,1, there is no need to scale the flow values together with resolution
        flow_tps_0 = F.interpolate(flow_tps, scale_factor=0.5)
        flow_tps_1 = F.interpolate(flow_tps_0, scale_factor=0.5)
        flow_tps_2 = F.interpolate(flow_tps_1, scale_factor=0.5)
        flow_tps_3 = F.interpolate(flow_tps_2, scale_factor=0.5)

        #feature pyramid to compute dense flow
        feature_T_3_tps = self.resample(feature_T_3.cuda(gpu_id), flow_tps_3, False).cuda(gpu_id)
        feature_T_2_tps = self.resample(feature_T_2.cuda(gpu_id), flow_tps_2, False).cuda(gpu_id)
        feature_T_1_tps = self.resample(feature_T_1.cuda(gpu_id), flow_tps_1, False).cuda(gpu_id)
        feature_T_0_tps = self.resample(feature_T_0.cuda(gpu_id), flow_tps_0, False).cuda(gpu_id)
        p3_T = self.feat_3_T(feature_T_3_tps)
        p2_T = self._upsample_add(p3_T, self.feat_2_T(feature_T_2_tps))
        p2_T = self.smooth_2_T(p2_T)
        p1_T = self._upsample_add(p2_T, self.feat_1_T(feature_T_1_tps))
        p1_T = self.smooth_1_T(p1_T)
        p0_T = self._upsample_add(p1_T, self.feat_0_T(feature_T_0_tps))
        p0_T = self.smooth_0_T(p0_T)

        p3_S = self.feat_3_S(feature_S_3)
        p2_S = self._upsample_add(p3_S, self.feat_2_S(feature_S_2))
        p2_S = self.smooth_2_S(p2_S)
        p1_S = self._upsample_add(p2_S, self.feat_1_S(feature_S_1))
        p1_S = self.smooth_1_S(p1_S)
        p0_S = self._upsample_add(p1_S, self.feat_0_S(feature_S_0))
        p0_S = self.smooth_0_S(p0_S)

        p3_P = self.feat_3_P(feature_prev_3)
        p2_P = self._upsample_add(p3_P, self.feat_2_P(feature_prev_2))
        p2_P = self.smooth_2_P(p2_P)
        p1_P = self._upsample_add(p2_P, self.feat_1_P(feature_prev_1))
        p1_P = self.smooth_1_P(p1_P)
        p0_P = self._upsample_add(p1_P, self.feat_0_P(feature_prev_0))
        p0_P = self.smooth_0_P(p0_P)
        
        warped_fg_tps = self.grid_sample(input_tfg, warped_grid.cuda(gpu_id)).cuda(gpu_id)

        if input_grid is not None:
            warped_grid_tps = self.grid_sample(input_grid, warped_grid.cuda(gpu_id)).cuda(gpu_id)

        warped_lo_tps = self.grid_sample(input_tlo, warped_grid.cuda(gpu_id)).cuda(gpu_id)

        feature_flow = torch.cat([p0_T, p0_S], dim=1)
        feature_flow = torch.cat([feature_flow, p0_P], dim=1)
        flow_dense = self.model_dense_flow(feature_flow)
        flow_tps = torch.cat([flow_tps[:, 0:1, :, :]*((width-1.0)/2.0), flow_tps[:, 1:2, :, :]*((height-1.0)/2.0)], dim=1)
        if not self.opt.tps_only:
            flow_total = flow_tps + flow_dense
        else:
            flow_total = flow_tps

        warped_lo_dense = self.resample(input_tlo, flow_total).cuda(gpu_id)

        warped_fg_dense = self.resample(input_tfg, flow_total).cuda(gpu_id)

        # uncomment for better quality during test
        #if not self.opt.isTrain:
        #    for i in range(1, self.opt.label_nc_2):
        #        R_avg = torch.sum(input_tfg[:,0][input_tlo[:, i] != 0] + 1) / (torch.sum(input_tlo[:, i] != 0) + 1e-6)
        #        G_avg = torch.sum(input_tfg[:,1][input_tlo[:, i] != 0] + 1) / (torch.sum(input_tlo[:, i] != 0) + 1e-6)
        #        B_avg = torch.sum(input_tfg[:,2][input_tlo[:, i] != 0] + 1) / (torch.sum(input_tlo[:, i] != 0) + 1e-6)
        #        if torch.isnan(R_avg) or torch.isnan(G_avg) or torch.isnan(B_avg):
        #            continue
        #        rand_value = random.uniform(-0.05,0.05)
        #        warped_fg_dense[:,0][(warped_lo_dense[:,i] == 0) & (input_slo[:,i] != 0)] = rand_value + R_avg - 1
        #        warped_fg_dense[:,1][(warped_lo_dense[:,i] == 0) & (input_slo[:,i] != 0)] = rand_value + G_avg - 1
        #        warped_fg_dense[:,2][(warped_lo_dense[:,i] == 0) & (input_slo[:,i] != 0)] = rand_value + B_avg - 1
        #    warped_lo_dense = input_slo


        return warped_fg_tps, warped_fg_dense, warped_lo_tps, warped_lo_dense, flow_tps, flow_dense, flow_total

##############################################################################
# Class for the Composition GAN of stage 3
##############################################################################
class Composer(BaseNetwork):
    def __init__(self, opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, n_blocks, use_fg_model=False, no_flow=False,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Composer, self).__init__()                
        self.opt = opt
        self.n_downsampling = n_downsampling
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        activation = nn.ReLU(True)
        
        ### flow and image generation
        ### downsample        
        model_down_T = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model_down_T += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), activation]  

        mult = 2**n_downsampling
        for i in range(n_blocks - n_blocks//2):
            model_down_T += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        model_down_S = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_S += copy.deepcopy(model_down_T[4:])
        model_down_fg = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_fg += copy.deepcopy(model_down_T[4:])
    
        ### resnet blocks
        model_res_fg = []
        for i in range(n_blocks//2):
            model_res_fg += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model_res_sdfl = copy.deepcopy(model_res_fg)      

        ### upsample
        model_up_fg = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up_fg += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm_layer(ngf*mult//2), activation]                    
        model_final_fg = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        model_up_sd = copy.deepcopy(model_up_fg)
        model_final_sd = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]

        if not no_flow:
            model_up_flow = copy.deepcopy(model_up_fg)
            model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]                
            model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()] 

        self.model_down_T = nn.Sequential(*model_down_T)        
        self.model_down_S = nn.Sequential(*model_down_S)        
        self.model_down_fg = nn.Sequential(*model_down_fg)        
        self.model_res_fg = nn.Sequential(*model_res_fg)
        self.model_res_sdfl = nn.Sequential(*model_res_sdfl)
        self.model_up_fg = nn.Sequential(*model_up_fg)
        self.model_up_sd = nn.Sequential(*model_up_sd)
        self.model_final_fg = nn.Sequential(*model_final_fg)
        self.model_final_sd = nn.Sequential(*model_final_sd)

        if not no_flow:
            self.model_up_flow = nn.Sequential(*model_up_flow)                
            self.model_final_flow = nn.Sequential(*model_final_flow)                       
            self.model_final_w = nn.Sequential(*model_final_w)

    def forward(self, input_T, input_S, input_SFG, input_BG, img_prev, use_raw_only):
        gpu_id = input_T.get_device()
        input_smask = input_S[:, -self.opt.label_nc_3:-self.opt.label_nc_3+1]
        input_smask_cloth = input_S[:, -self.opt.label_nc_3+1:-self.opt.label_nc_3+self.opt.label_nc_2].sum(dim=1, keepdim=True)
        
        input_SFG[:,0][(input_smask_cloth[:,0]==0)] = -1
        input_SFG[:,1][(input_smask_cloth[:,0]==0)] = -1
        input_SFG[:,2][(input_smask_cloth[:,0]==0)] = -1
        
        input_S_full = torch.cat([input_S, input_SFG], dim=1)
        downsample_1 = self.model_down_T(input_T)
        downsample_2 = self.model_down_S(input_S_full)
        downsample_3 = self.model_down_fg(img_prev)
        fg_feat = self.model_up_fg(self.model_res_fg(downsample_1+downsample_2+downsample_3))
        res_sdfl = self.model_res_sdfl(downsample_2+downsample_3)
        sd_feat = self.model_up_sd(res_sdfl)
        fg_res = self.model_final_fg(fg_feat)
        
        fg = (1-input_smask_cloth).expand_as(fg_res) * (fg_res + 1) + input_SFG
        #not sure if it is appropriate

        fg[:,0][input_smask[:,0]==1] = -1
        fg[:,1][input_smask[:,0]==1] = -1
        fg[:,2][input_smask[:,0]==1] = -1

        sd = self.model_final_sd(sd_feat)
        BG = torch.zeros_like(input_BG).cuda(gpu_id)
        BG = sd.expand_as(input_BG) * (input_BG + 1)
        img_raw = fg + BG 

        flow = weight = flow_feat = None
        if not self.no_flow:
            flow_feat = self.model_up_flow(res_sdfl)                                                              
            flow = self.model_final_flow(flow_feat) * 20
            weight = self.model_final_w(flow_feat) 
        if self.no_flow:
            img_final = img_raw
        else:
            img_warp = self.resample(img_prev[:,-3:,...].cuda(gpu_id), flow).cuda(gpu_id)        
            weight_ = weight.expand_as(img_raw)
            img_final = img_raw * weight_ + img_warp * (1-weight_)             

        return img_final, img_raw, fg_res, sd, fg, flow, weight

##############################################################################
# Class for the Layout GAN of stage 1
##############################################################################
class Parser(BaseNetwork):
    def __init__(self, opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, n_blocks, use_fg_model=False, no_flow=False,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Parser, self).__init__()                
        self.opt = opt
        self.n_downsampling = n_downsampling
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        activation = nn.ReLU(True)
        
        ### flow and image generation
        ### downsample        
        model_down_T = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model_down_T += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), activation]  

        mult = 2**n_downsampling
        for i in range(n_blocks - n_blocks//2):
            model_down_T += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        model_down_S = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_S += copy.deepcopy(model_down_T[4:])
        model_down_lo = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_lo += copy.deepcopy(model_down_T[4:])
    
        ### resnet blocks
        model_res_lo = []
        for i in range(n_blocks//2):
            model_res_lo += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        model_up_lo = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up_lo += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm_layer(ngf*mult//2), activation]                    
        model_final_lo = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_final_softmax = [nn.Softmax(dim=1)]
        #model_final_logsoftmax = [nn.LogSoftmax(dim=1)]

        model_res_flow = copy.deepcopy(model_res_lo)
        model_up_flow = copy.deepcopy(model_up_lo)
        model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]                
        model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()] 

        self.model_down_T = nn.Sequential(*model_down_T)        
        self.model_down_S = nn.Sequential(*model_down_S)        
        self.model_down_lo = nn.Sequential(*model_down_lo)        
        self.model_res_lo = nn.Sequential(*model_res_lo)
        self.model_up_lo = nn.Sequential(*model_up_lo)
        self.model_final_lo = nn.Sequential(*model_final_lo)
        self.model_final_softmax = nn.Sequential(*model_final_softmax)
        #self.model_final_logsoftmax = nn.Sequential(*model_final_logsoftmax)
        self.model_res_flow = nn.Sequential(*model_res_flow)
        self.model_up_flow = nn.Sequential(*model_up_flow)
        self.model_final_flow = nn.Sequential(*model_final_flow)
        self.model_final_w = nn.Sequential(*model_final_w)

    def forward(self, input_T, input_S, lo_prev, use_raw_only):
        gpu_id = input_T.get_device()
        downsample_1 = self.model_down_T(input_T)
        downsample_2 = self.model_down_S(input_S)
        downsample_3 = self.model_down_lo(lo_prev)
        lo_feat = self.model_up_lo(self.model_res_lo(downsample_1+downsample_2+downsample_3))
        lo_raw = self.model_final_lo(lo_feat)
        lo_softmax_raw = self.model_final_softmax(lo_raw)
        lo_logsoftmax_raw = torch.log(torch.abs(lo_softmax_raw)+1e-6)

        flow_feat = self.model_up_flow(self.model_res_flow(downsample_2+downsample_3))
        flow = self.model_final_flow(flow_feat) * 20
        weight = self.model_final_w(flow_feat)

        lo_softmax_warp = self.resample(lo_prev[:,-self.opt.label_nc_1:,...].cuda(gpu_id), flow).cuda(gpu_id)        
        weight_ = weight.expand_as(lo_softmax_raw)
        lo_softmax_final = lo_softmax_raw * weight_ + lo_softmax_warp * (1-weight_) 
        lo_logsoftmax_final = torch.log(torch.abs(lo_softmax_final)+1e-6)

        return lo_softmax_final, lo_softmax_raw, lo_logsoftmax_final, lo_logsoftmax_raw, flow, weight

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetBlock_deform(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock_deform, self).__init__()
        self.conv_block_1 = self.build_conv_block_1(dim, padding_type, norm_layer, activation, use_dropout)
        self.conv_block_2 = self.build_conv_block_2(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block_1(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [ModulatedDeformConvPack(dim, dim, kernel_size=(3,3), stride=1, padding=p, deformable_groups=1, bias=True),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def build_conv_block_2(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [ModulatedDeformConvPack(dim, dim, kernel_size=(3,3), stride=1, padding=p, deformable_groups=1, bias=True),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, input_list):
        x, input_LO = input_list
        res = self.conv_block_1([x, input_LO, input_LO])
        res = self.conv_block_2([res, input_LO, input_LO])
        out = x + res
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, norm_layer,
                                       getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]            
            for i in range(len(model)):
                result.append(model[i](result[-1]))            
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))                                
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)                    
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)            

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, label_nc):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss2d()

    def forward(self, output, label):
        label = label.long().max(1)[1]        
        output = self.softmax(output)
        return self.criterion(output, label)

class PixelwiseSoftmaxLoss(nn.Module):
    def __init__(self):
        super(PixelwiseSoftmaxLoss, self).__init__()
        self.criterion = nn.NLLLoss2d()

    def forward(self, output, label):
        label = label.long().max(1)[1]
        return self.criterion(output, label)

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):        
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask, target * mask)
        return loss

class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        #self.weights = [0.5, 1, 2, 8, 32]
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, input, target, mask=None):
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, input.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:                
                loss += self.weights[i] * self.criterion(input * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights)-1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss

from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,flow):
        batch_size = flow.size()[0]
        h_x = flow.size()[2]
        w_x = flow.size()[3]
        count_h = self._tensor_size(flow[:,:,1:,:])
        count_w = self._tensor_size(flow[:,:,:,1:])
        h_tv = torch.abs((flow[:,:,1:,:]-flow[:,:,:h_x-1,:])).sum()
        w_tv = torch.abs((flow[:,:,:,1:]-flow[:,:,:,:w_x-1])).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]