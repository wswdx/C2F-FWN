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

        input_nc_T_3 = opt.input_nc_T_3
        input_nc_S_3 = opt.input_nc_S_3 * opt.n_frames_G + 3
        prev_output_nc = (opt.n_frames_G - 1) * opt.output_nc_3    

        self.netG = networks.define_composer(input_nc_T_3, input_nc_S_3, opt.output_nc_3, prev_output_nc, opt.ngf, 
                                       opt.n_downsample_G, opt.norm, 0, self.gpu_ids, opt)

        print('---------- Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:                    
            self.load_network(self.netG, 'Composer', opt.which_epoch, opt.load_pretrain)
                        
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

    def encode_input(self, input_TParsing, input_TFG, input_SPose, input_SParsing, input_SFG, input_SFG_full, input_BG, input_SI):        
        size = input_SPose.size()
        self.bs, tG, self.height, self.width = size[0], size[1], size[3], size[4]
        
        input_TParsing = input_TParsing.data.cuda()
        input_TFG = input_TFG.data.cuda()
        input_SPose = input_SPose.data.cuda()
        input_SParsing = input_SParsing.data.cuda()
        input_SFG = Variable(input_SFG.data.cuda())
        if input_SFG_full is not None:
            input_SFG_full = Variable(input_SFG_full.data.cuda())
        input_BG = input_BG.data.cuda()
        input_BG[:,:,0][input_SParsing[:,:,0] != 0] = -1
        input_BG[:,:,1][input_SParsing[:,:,0] != 0] = -1
        input_BG[:,:,2][input_SParsing[:,:,0] != 0] = -1
        input_BG = Variable(input_BG)
        if input_SI is not None:
            input_SI = Variable(input_SI.data.cuda())

        oneHot_size = (self.bs, 1, self.opt.label_nc_3-self.opt.label_nc_2+1, self.height, self.width)
        TParsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        TParsing_label = TParsing_label.scatter_(2, input_TParsing.long(), 1.0)
        input_TParsing = TParsing_label
        input_TParsing = Variable(input_TParsing)

        oneHot_size = (self.bs, tG, self.opt.label_nc_3, self.height, self.width)
        SParsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        SParsing_label = SParsing_label.scatter_(2, input_SParsing.long(), 1.0)
        input_SParsing = SParsing_label
        input_SParsing = Variable(input_SParsing)

        input_1 = torch.cat([input_TParsing, input_TFG], dim=2)
        input_2 = torch.cat([input_SPose, input_SParsing], dim=2)
        
        return input_1, input_2, input_SFG, input_BG, input_SFG_full, input_SI

    def forward(self, input_TParsing, input_TFG, input_SPose, input_SParsing, input_SFG, input_BG, input_SFG_full, input_SI, fake_SI_prev, dummy_bs=0):
        tG = self.opt.n_frames_G           
        gpu_split_id = self.opt.n_gpus_gen + 1        

        real_input_1, real_input_2, real_input_SFG, real_input_BG, real_SFG_full, real_SI = self.encode_input(input_TParsing, input_TFG, input_SPose, input_SParsing, input_SFG, input_SFG_full, input_BG, input_SI)        

        is_first_frame = fake_SI_prev is None
        if is_first_frame: # at the beginning of a sequence; needs to generate the first frame
            fake_SI_prev = Variable(self.Tensor(self.bs, tG-1, self.opt.output_nc_3, self.height, self.width).zero_())

        netG = torch.nn.parallel.replicate(self.netG, self.opt.gpu_ids[:gpu_split_id]) if self.split_gpus else self.netG
        start_gpu = self.gpu_ids[1] if self.split_gpus else real_input_1.get_device()        
        fake_SI, fake_SI_raw, fake_sd, fake_SFG_full, fake_SFG_res, flow, weight = self.generate_frame_train(netG, real_input_1, real_input_2, real_input_SFG, real_input_BG, fake_SI_prev, start_gpu, is_first_frame)        
        fake_SI_prev = fake_SI[:, -tG+1:].detach()
        fake_SI = fake_SI[:, tG-1:]

        return fake_SI, fake_SI_raw, fake_sd, fake_SFG_full, fake_SFG_res, flow, weight, real_input_1, real_input_2[:,tG-1:], real_input_SFG[:,tG-1:], real_input_BG[:,tG-1:], real_SI[:,tG-2:], real_SFG_full[:,tG-2:], fake_SI_prev

    def generate_frame_train(self, netG, real_input_1, real_input_2, real_input_SFG, real_input_BG, fake_SI_prev, start_gpu, is_first_frame):        
        tG = self.opt.n_frames_G        
        n_frames_load = self.n_frames_load
        dest_id = self.gpu_ids[0] if self.split_gpus else start_gpu        

        ### generate inputs      
        fake_SIs_raw, fake_sds, fake_SFGs_full, fake_SFGs_res, flows, weights = None, None, None, None, None, None       
        
        ### sequentially generate each frame
        for t in range(n_frames_load):
            gpu_id = (t // self.n_frames_per_gpu + start_gpu) if self.split_gpus else start_gpu # the GPU idx where we generate this frame
            net_id = gpu_id if self.split_gpus else 0                                           # the GPU idx where the net is located

            real_input_1_reshaped = real_input_1[:, 0].cuda(gpu_id)            
            real_input_2_reshaped = real_input_2[:, t:t+tG,...].view(self.bs, -1, self.height, self.width).cuda(gpu_id)
            real_input_BG_reshaped = real_input_BG[:, t+tG-1:t+tG,...].view(self.bs, -1, self.height, self.width).cuda(gpu_id)
            real_input_SFG_reshaped = real_input_SFG[:, t+tG-1:t+tG,...].view(self.bs, -1, self.height, self.width).cuda(gpu_id)

            fake_SI_prevs = fake_SI_prev[:, t:t+tG-1,...].cuda(gpu_id)
            if (t % self.n_frames_bp) == 0:
                fake_SI_prevs = fake_SI_prevs.detach()
            fake_SI_prevs_reshaped = fake_SI_prevs.view(self.bs, -1, self.height, self.width)
            
            use_raw_only = self.opt.no_first_img and is_first_frame 
            fake_SI, fake_SI_raw, fake_SFG_res, fake_sd, fake_SFG_full, flow, weight \
                = netG.forward(real_input_1_reshaped, real_input_2_reshaped, real_input_SFG_reshaped, real_input_BG_reshaped, fake_SI_prevs_reshaped, use_raw_only)

            fake_SI_prev = self.concat([fake_SI_prev, fake_SI.unsqueeze(1).cuda(dest_id)], dim=1)
            fake_SIs_raw = self.concat([fake_SIs_raw, fake_SI_raw.unsqueeze(1).cuda(dest_id)], dim=1)
            fake_sds = self.concat([fake_sds, fake_sd.unsqueeze(1).cuda(dest_id)], dim=1)
            fake_SFGs_full = self.concat([fake_SFGs_full, fake_SFG_full.unsqueeze(1).cuda(dest_id)], dim=1)
            fake_SFGs_res = self.concat([fake_SFGs_res, fake_SFG_res.unsqueeze(1).cuda(dest_id)], dim=1)
            if flow is not None:
                flows = self.concat([flows, flow.unsqueeze(1).cuda(dest_id)], dim=1)
                weights = self.concat([weights, weight.unsqueeze(1).cuda(dest_id)], dim=1)                    
       
        return fake_SI_prev, fake_SIs_raw, fake_sds, fake_SFGs_full, fake_SFGs_res, flows, weights

    def inference(self, input_TParsing, input_TFG, input_SPose, input_SParsing, input_SFG, input_BG, input_SFG_full=None, input_SI=None):
        with torch.no_grad():
            real_input_1, real_input_2, real_input_SFG, real_input_BG, real_SFG_full, real_SI \
             = self.encode_input(input_TParsing, input_TFG, input_SPose, input_SParsing, input_SFG, input_SFG_full, input_BG, input_SI)            
            self.is_first_frame = not hasattr(self, 'fake_SI_prev') or self.fake_SI_prev is None
            if self.is_first_frame:
                self.fake_SI_prev = Variable(self.Tensor(self.bs, self.opt.n_frames_G-1, self.opt.output_nc_3, self.height, self.width).zero_())
            
            fake_SI, fake_SI_raw, fake_SFG_res, fake_sd, fake_SFG_full, flow, weight = self.generate_frame_infer(real_input_1, real_input_2, real_input_SFG, real_input_BG)
        if input_SFG_full is not None and input_SI is not None:
            return fake_SI, fake_SI_raw, fake_SFG_res, fake_sd, fake_SFG_full, flow, weight, real_input_1[0, -1], real_input_2[0, -1], real_input_SFG[0, -1], real_SFG_full[0, -1], real_SI[0, -1]
        else:
            return fake_SI, fake_SI_raw, fake_SFG_res, fake_sd, fake_SFG_full, real_input_1[0, -1], real_input_2[0, -1]

    def generate_frame_infer(self, real_input_1, real_input_2, real_input_SFG, real_input_BG):
        tG = self.opt.n_frames_G
        netG = self.netG
        
        ### prepare inputs
        real_input_1_reshaped = real_input_1[0, 0:1]
        real_input_2_reshaped = real_input_2[0, :tG].view(1, -1, self.height, self.width)
        if real_input_SFG.size()[1] > 1:
            real_input_SFG_reshaped = real_input_SFG[0, tG-1:tG].view(1, -1, self.height, self.width)
        else:
            real_input_SFG_reshaped = real_input_SFG[0, 0:1].view(1, -1, self.height, self.width)

        real_input_BG_reshaped = real_input_BG[0, tG-1:tG].view(1, -1, self.height, self.width)
        fake_SI_prev_reshaped = self.fake_SI_prev.view(1, -1, self.height, self.width)               
        use_raw_only = self.opt.no_first_img and self.is_first_frame

        ### network forward        
        fake_SI, fake_SI_raw, fake_SFG_res, fake_sd, fake_SFG_full, flow, weight \
            = netG.forward(real_input_1_reshaped, real_input_2_reshaped, real_input_SFG_reshaped, real_input_BG_reshaped, fake_SI_prev_reshaped, use_raw_only)    

        self.fake_SI_prev = torch.cat([self.fake_SI_prev[:,1:], fake_SI.unsqueeze(1)], dim=1)        
        return fake_SI, fake_SI_raw, fake_SFG_res, fake_sd, fake_SFG_full, flow, weight

    def generate_first_frame(self, real_A, real_B, pool_map=None):
        tG = self.opt.n_frames_G
        if self.opt.no_first_img:          # model also generates first frame            
            fake_B_prev = Variable(self.Tensor(self.bs, tG-1, self.opt.output_nc_3, self.height, self.width).zero_())
        elif self.opt.isTrain or self.opt.use_real_img: # assume first frame is given
            fake_B_prev = real_B[:,:(tG-1),...]            
        elif self.opt.use_single_G:        # use another model (trained on single images) to generate first frame
            fake_B_prev = None
            if self.opt.use_instance:
                real_A = real_A[:,:,:self.opt.label_nc_3,:,:]
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
        self.save_network(self.netG, 'Composer', label, self.gpu_ids)                    