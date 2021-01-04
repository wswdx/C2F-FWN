### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import math
import torch
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks

class Vid2VidModelG(BaseModel):
    def name(self):
        return 'Vid2VidModelG'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain        
        if not opt.debug:
            torch.backends.cudnn.benchmark = True       
        
        # define net G                        
        self.n_scales = opt.n_scales_spatial        
        self.split_gpus = (self.opt.n_gpus_gen < len(self.opt.gpu_ids)) and (self.opt.batchSize == 1)

        input_nc_T_1 = opt.input_nc_T_1
        input_nc_S_1 = opt.input_nc_S_1 * opt.n_frames_G
        prev_output_nc = (opt.n_frames_G - 1) * opt.label_nc_1

        self.netG = networks.define_parser(input_nc_T_1, input_nc_S_1, opt.label_nc_1, prev_output_nc, opt.ngf, 
                                       opt.n_downsample_G, opt.norm, 0, self.gpu_ids, opt)

        print('---------- Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:       
            self.load_network(self.netG, 'Parser', opt.which_epoch, opt.load_pretrain)

        # define training variables
        if self.isTrain:            
            self.n_gpus = self.opt.n_gpus_gen if self.opt.batchSize == 1 else 1    # number of gpus for running generator            
            self.n_frames_bp = 1                                                   # number of frames to backpropagate the loss            
            self.n_frames_per_gpu = min(self.opt.max_frames_per_gpu, self.opt.n_frames_total // self.n_gpus) # number of frames in each GPU
            self.n_frames_load = self.n_gpus * self.n_frames_per_gpu   # number of frames in all GPUs            
            if self.opt.debug:
                print('training %d frames at once, using %d gpus, frames per gpu = %d' % (self.n_frames_load, 
                    self.n_gpus, self.n_frames_per_gpu))

        # set loss functions and optimizers
        if self.isTrain:                      
            self.old_lr = opt.lr
            # initialize optimizer G
            params = list(self.netG.parameters())

            if opt.TTUR:                
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr            
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

    def encode_input(self, input_TParsing, input_SPose, input_SParsing=None, input_SFG=None):        
        size = input_SPose.size()
        self.bs, tG, self.height, self.width = size[0], size[1], size[3], size[4]
        
        input_TParsing = input_TParsing.data.cuda()
        input_SPose = input_SPose.data.cuda()
        if input_SFG is not None:
            input_SFG = input_SFG.data.cuda()

        oneHot_size = (self.bs, 1, self.opt.label_nc_1, self.height, self.width)
        TParsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        TParsing_label = TParsing_label.scatter_(2, input_TParsing.long(), 1.0)
        input_TParsing = TParsing_label
        input_TParsing = Variable(input_TParsing)

        if input_SParsing is not None:
            input_SParsing = input_SParsing.data.cuda()
            oneHot_size = (self.bs, tG, self.opt.label_nc_1, self.height, self.width)
            SParsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            SParsing_label = SParsing_label.scatter_(2, input_SParsing.long(), 1.0)
            input_SParsing = SParsing_label
            input_SParsing = Variable(input_SParsing)

        return input_TParsing, input_SPose, input_SParsing, input_SFG

    def forward(self, input_TParsing, input_SPose, input_SParsing, input_SFG, fake_slo_prev, dummy_bs=0):
        tG = self.opt.n_frames_G           
        gpu_split_id = self.opt.n_gpus_gen + 1        

        real_input_T, real_input_S, real_slo, real_sfg = self.encode_input(input_TParsing, input_SPose, input_SParsing, input_SFG)        

        is_first_frame = fake_slo_prev is None
        if is_first_frame: # at the beginning of a sequence; needs to generate the first frame
            fake_slo_prev = Variable(self.Tensor(self.bs, tG-1, self.opt.label_nc_1, self.height, self.width).zero_())

        netG = torch.nn.parallel.replicate(self.netG, self.opt.gpu_ids[:gpu_split_id]) if self.split_gpus else self.netG
        start_gpu = self.gpu_ids[1] if self.split_gpus else real_input_T.get_device()        
        fake_slo, fake_slo_raw, fake_slo_ls, fake_slo_raw_ls, flow, weight = self.generate_frame_train(netG, real_input_T, real_input_S, fake_slo_prev, start_gpu, is_first_frame)        
        fake_slo_prev = fake_slo[:, -tG+1:].detach()
        fake_slo = fake_slo[:, tG-1:]

        return fake_slo, fake_slo_raw, fake_slo_ls, fake_slo_raw_ls, flow, weight, real_input_T, real_input_S[:,tG-1:], real_slo[:,tG-2:], real_sfg[:,tG-2:], fake_slo_prev

    def generate_frame_train(self, netG, real_input_T, real_input_S, fake_slo_prev, start_gpu, is_first_frame):        
        tG = self.opt.n_frames_G        
        n_frames_load = self.n_frames_load
        dest_id = self.gpu_ids[0] if self.split_gpus else start_gpu        

        ### generate inputs              
        fake_slos_raw, fake_slos_ls, fake_slos_raw_ls, flows, weights = None, None, None, None, None
        ### sequentially generate each frame
        for t in range(n_frames_load):
            gpu_id = (t // self.n_frames_per_gpu + start_gpu) if self.split_gpus else start_gpu # the GPU idx where we generate this frame
            net_id = gpu_id if self.split_gpus else 0                                           # the GPU idx where the net is located

            real_input_T_reshaped = real_input_T[:, 0].cuda(gpu_id)            
            real_input_S_reshaped = real_input_S[:, t:t+tG,...].view(self.bs, -1, self.height, self.width).cuda(gpu_id)
 
            fake_slo_prevs = fake_slo_prev[:, t:t+tG-1,...].cuda(gpu_id)
            if (t % self.n_frames_bp) == 0:
                fake_slo_prevs = fake_slo_prevs.detach()
            fake_slo_prevs_reshaped = fake_slo_prevs.view(self.bs, -1, self.height, self.width)
            
            use_raw_only = self.opt.no_first_img and is_first_frame 
            fake_slo, fake_slo_raw, fake_slo_ls, fake_slo_raw_ls, flow, weight = netG.forward(real_input_T_reshaped, real_input_S_reshaped, fake_slo_prevs_reshaped, use_raw_only)

            fake_slo_prev = self.concat([fake_slo_prev, fake_slo.unsqueeze(1).cuda(dest_id)], dim=1)               
            fake_slos_raw = self.concat([fake_slos_raw, fake_slo_raw.unsqueeze(1).cuda(dest_id)], dim=1)
            fake_slos_ls = self.concat([fake_slos_ls, fake_slo_ls.unsqueeze(1).cuda(dest_id)], dim=1)
            fake_slos_raw_ls = self.concat([fake_slos_raw_ls, fake_slo_raw_ls.unsqueeze(1).cuda(dest_id)], dim=1)
            flows = self.concat([flows, flow.unsqueeze(1).cuda(dest_id)], dim=1)
            weights = self.concat([weights, weight.unsqueeze(1).cuda(dest_id)], dim=1)

        return fake_slo_prev, fake_slos_raw, fake_slos_ls, fake_slos_raw_ls, flows, weights

    def inference(self, input_TParsing, input_SPose, input_SParsing=None):
        with torch.no_grad():
            real_input_T, real_input_S, real_slo, _ \
             = self.encode_input(input_TParsing, input_SPose, input_SParsing)            
            self.is_first_frame = not hasattr(self, 'fake_slo_prev') or self.fake_slo_prev is None
            if self.is_first_frame:
                self.fake_slo_prev = Variable(self.Tensor(self.bs, self.opt.n_frames_G-1, self.opt.label_nc_1, self.height, self.width).zero_())
            
            fake_slo = self.generate_frame_infer(real_input_T, real_input_S)

        if real_slo is not None:
            return fake_slo, real_input_T[0, -1], real_input_S[0, -1], real_slo[0, -1]
        else:
            return fake_slo, real_input_T[0, -1], real_input_S[0, -1]

    def generate_frame_infer(self, real_input_T, real_input_S):
        tG = self.opt.n_frames_G
        netG = self.netG
        
        ### prepare inputs
        real_input_T_reshaped = real_input_T[0, 0:1]
        real_input_S_reshaped = real_input_S[0, :tG].view(1, -1, self.height, self.width)
        fake_slo_prev_reshaped = self.fake_slo_prev.view(1, -1, self.height, self.width)               
        use_raw_only = self.opt.no_first_img and self.is_first_frame

        ### network forward        
        fake_slo, _, _, _, _, _ = netG.forward(real_input_T_reshaped, real_input_S_reshaped, fake_slo_prev_reshaped, use_raw_only)    

        self.fake_slo_prev = torch.cat([self.fake_slo_prev[:,1:], fake_slo.unsqueeze(1)], dim=1)        
        return fake_slo

    def generate_first_frame(self, real_A, real_B, pool_map=None):
        tG = self.opt.n_frames_G
        if self.opt.no_first_img:          # model also generates first frame            
            fake_B_prev = Variable(self.Tensor(self.bs, tG-1, self.opt.output_nc_2, self.height, self.width).zero_())
        elif self.opt.isTrain or self.opt.use_real_img: # assume first frame is given
            fake_B_prev = real_B[:,:(tG-1),...]            
        elif self.opt.use_single_G:        # use another model (trained on single images) to generate first frame
            fake_B_prev = None
            if self.opt.use_instance:
                real_A = real_A[:,:,:self.opt.label_nc_2,:,:]
            for i in range(tG-1):                
                feat_map = self.get_face_features(real_B[:,i], pool_map[:,i]) if self.opt.dataset_mode == 'face' else None
                fake_B = self.netG_i.forward(real_A[:,i], feat_map).unsqueeze(1)                
                fake_B_prev = self.concat([fake_B_prev, fake_B], dim=1)
        else:
            raise ValueError('Please specify the method for generating the first frame')
            
        fake_B_prev = self.build_pyr(fake_B_prev)
        if not self.opt.isTrain:
            fake_B_prev = [B[0] for B in fake_B_prev]
        return fake_B_prev    

    def return_dummy(self, input_A):
        h, w = input_A.size()[3:]
        t = self.n_frames_load
        tG = self.opt.n_frames_G  
        flow, weight = (self.Tensor(1, t, 2, h, w), self.Tensor(1, t, 1, h, w)) if not self.opt.no_flow else (None, None)
        return self.Tensor(1, t, 3, h, w), self.Tensor(1, t, 3, h, w), flow, weight, \
               self.Tensor(1, t, self.opt.input_nc, h, w), self.Tensor(1, t+1, 3, h, w), self.build_pyr(self.Tensor(1, tG-1, 3, h, w))

    def load_single_G(self): # load the model that generates the first frame
        opt = self.opt     
        s = self.n_scales
        if 'City' in self.opt.dataroot:
            single_path = 'checkpoints/label2city_single/'
            if opt.loadSize == 512:
                load_path = single_path + 'latest_net_G_512.pth'            
                netG = networks.define_G(35, 3, 0, 64, 'global', 3, 'instance', 0, self.gpu_ids, opt)                
            elif opt.loadSize == 1024:                            
                load_path = single_path + 'latest_net_G_1024.pth'
                netG = networks.define_G(35, 3, 0, 64, 'global', 4, 'instance', 0, self.gpu_ids, opt)                
            elif opt.loadSize == 2048:     
                load_path = single_path + 'latest_net_G_2048.pth'
                netG = networks.define_G(35, 3, 0, 32, 'local', 4, 'instance', 0, self.gpu_ids, opt)
            else:
                raise ValueError('Single image generator does not exist')
        elif 'face' in self.opt.dataroot:            
            single_path = 'checkpoints/edge2face_single/'
            load_path = single_path + 'latest_net_G.pth' 
            opt.feat_num = 16           
            netG = networks.define_G(15, 3, 0, 64, 'global_with_features', 3, 'instance', 0, self.gpu_ids, opt)
            encoder_path = single_path + 'latest_net_E.pth'
            self.netE = networks.define_G(3, 16, 0, 16, 'encoder', 4, 'instance', 0, self.gpu_ids)
            self.netE.load_state_dict(torch.load(encoder_path))
        else:
            raise ValueError('Single image generator does not exist')
        netG.load_state_dict(torch.load(load_path))        
        return netG

    def get_face_features(self, real_image, inst):                
        feat_map = self.netE.forward(real_image, inst)            
        #if self.opt.use_encoded_image:
        #    return feat_map
        
        load_name = 'checkpoints/edge2face_single/features.npy'
        features = np.load(load_name, encoding='latin1').item()                        
        inst_np = inst.cpu().numpy().astype(int)

        # find nearest neighbor in the training dataset
        num_images = features[6].shape[0]
        feat_map = feat_map.data.cpu().numpy()
        feat_ori = torch.FloatTensor(7, self.opt.feat_num, 1) # feature map for test img (for each facial part)
        feat_ref = torch.FloatTensor(7, self.opt.feat_num, num_images) # feature map for training imgs
        for label in np.unique(inst_np):
            idx = (inst == int(label)).nonzero() 
            for k in range(self.opt.feat_num): 
                feat_ori[label,k] = float(feat_map[idx[0,0], idx[0,1] + k, idx[0,2], idx[0,3]])
                for m in range(num_images):
                    feat_ref[label,k,m] = features[label][m,k]                
        cluster_idx = self.dists_min(feat_ori.expand_as(feat_ref).cuda(), feat_ref.cuda(), num=1)

        # construct new feature map from nearest neighbors
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for label in np.unique(inst_np):
            feat = features[label][:,:-1]                                                    
            idx = (inst == int(label)).nonzero()                
            for k in range(self.opt.feat_num):                    
                feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[min(cluster_idx, feat.shape[0]-1), k]
        
        return Variable(feat_map)

    def compute_mask(self, real_As, ts, te=None): # compute the mask for foreground objects
        _, _, _, h, w = real_As.size() 
        if te is None:
            te = ts + 1        
        mask_F = real_As[:, ts:te, self.opt.fg_labels[0]].clone()
        for i in range(1, len(self.opt.fg_labels)):
            mask_F = mask_F + real_As[:, ts:te, self.opt.fg_labels[i]]
        mask_F = torch.clamp(mask_F, 0, 1)
        return mask_F    

    def compute_fake_B_prev(self, real_B_prev, fake_B_last, fake_B):
        fake_B_prev = real_B_prev[:, 0:1] if fake_B_last is None else fake_B_last[:, -1:]
        if fake_B.size()[1] > 1:
            fake_B_prev = torch.cat([fake_B_prev, fake_B[:, :-1].detach()], dim=1)
        return fake_B_prev

    def save(self, label):        
        self.save_network(self.netG, 'Parser', label, self.gpu_ids)                    