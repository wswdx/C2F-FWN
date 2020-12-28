import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_transform_fixed, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

# dataset for our C2F-FWN of stage 2
class ClothDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 
        
        self.dir_tparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/target')
        self.dir_timg = os.path.join(opt.dataroot, opt.phase + '_img/target')
        self.dir_sparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/source')
        self.dir_simg = os.path.join(opt.dataroot, opt.phase + '_img/source')
        self.dir_grid = os.path.join(opt.dataroot, opt.phase + '_grid')
        self.tparsing_paths = sorted(make_grouped_dataset(self.dir_tparsing))
        self.timg_paths = sorted(make_grouped_dataset(self.dir_timg))
        self.sparsing_paths = sorted(make_grouped_dataset(self.dir_sparsing))
        self.simg_paths = sorted(make_grouped_dataset(self.dir_simg))

        self.init_frame_idx_cloth(self.simg_paths)

    def __getitem__(self, index):
        TParsing, TFG, SParsing, SFG, SFG_full, seq_idx = self.update_frame_idx_cloth(self.simg_paths, index)
        simg_paths = self.simg_paths[seq_idx]        
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(simg_paths), self.frame_idx)
        
        simg = Image.open(simg_paths[start_idx]).convert('RGB')     
        size = simg.size

        BigSizeFlag = True
        if size[0]/size[1] > 1:
            BigSizeFlag = True
        else:
            BigSizeFlag = False

        if BigSizeFlag:
            params = get_img_params(self.opt, (1920,1080))
        else:
            params = get_img_params(self.opt, size)

        tparsing_path = self.tparsing_paths[seq_idx][0]
        timg_path = self.timg_paths[seq_idx][0]

        TParsing, TFG = self.get_TImage(tparsing_path, timg_path, size, params, BigSizeFlag)
        TParsing, TFG = self.crop(TParsing), self.crop(TFG)

        frame_range = list(range(n_frames_total)) if (self.opt.isTrain or self.TParsing is None) else [self.opt.n_frames_G-1]
        for i in frame_range:
            simg_path = simg_paths[start_idx + i * t_step]
            sparsing_path = self.sparsing_paths[seq_idx][start_idx + i * t_step]

            SParsing_i, SFG_i, SFG_full_i = self.get_SImage(sparsing_path, simg_path, size, params, BigSizeFlag)

            SParsing_i = self.crop(SParsing_i)
            SFG_i, SFG_full_i = self.crop(SFG_i), self.crop(SFG_full_i)

            SParsing = concat_frame(SParsing, SParsing_i, n_frames_total)
            SFG = concat_frame(SFG, SFG_i, n_frames_total)
            SFG_full = concat_frame(SFG_full, SFG_full_i, n_frames_total)
        
        if not self.opt.isTrain:
            self.TParsing, self.TFG, self.SParsing, self.SFG, self.SFG_full = TParsing, TFG, SParsing, SFG, SFG_full
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'TParsing': TParsing, 'TFG': TFG, 'SParsing': SParsing, 'SFG': SFG, 'SFG_full': SFG_full, 'A_path': simg_path, 'change_seq': change_seq}
        return return_list

    def get_SImage(self, sparsing_path, simg_path, size, params, BigSizeFlag):          
        SI = Image.open(simg_path).convert('RGB')
        if SI.size != (1920,1080) and BigSizeFlag:
            SI = SI.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag:
            SI = SI.resize((192,256), Image.BICUBIC)
        SFG_np = np.array(SI)

        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0

        SParsing = Image.open(sparsing_path)
        SParsing_size = SParsing.size
        if SParsing_size != (1920,1080) and SParsing_size != (192,256) and BigSizeFlag:
            SParsing = SParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag and SParsing_size != (192,256):
            SParsing = SParsing.resize((192,256), Image.NEAREST)
        SParsing_np = np.array(SParsing)

        if SParsing_size == (192,256):
            SParsing_new_np = SParsing_np
            SParsing_new_np[(SParsing_np != 1) & (SParsing_np != 2)] = 0
        else:
            SParsing_new_np = np.zeros_like(SParsing_np)
            SParsing_new_np[(SParsing_np == 3) | (SParsing_np == 5) | (SParsing_np == 6) | (SParsing_np == 7) | (SParsing_np == 11)] = 1
            SParsing_new_np[(SParsing_np == 8) | (SParsing_np == 9) | (SParsing_np == 12)] = 2

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)

        SParsing_new = Image.fromarray(SParsing_new_np)
        if SParsing_size != (192,256) and BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        SParsing_scaled = transform_scale(SParsing_new)*255.0

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        SFG_full_scaled = transform_scale(Image.fromarray(SFG_np))
        SFG_np[:,:,0][(SParsing_new_np == 0)] = 0
        SFG_np[:,:,1][(SParsing_new_np == 0)] = 0
        SFG_np[:,:,2][(SParsing_new_np == 0)] = 0
        SFG_scaled = transform_scale(Image.fromarray(SFG_np))

        return SParsing_scaled, SFG_scaled, SFG_full_scaled

    def get_TImage(self, tparsing_path, timg_path, size, params, BigSizeFlag):          
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        TI = Image.open(timg_path).convert('RGB')

        if TI.size != (1920,1080) and BigSizeFlag:
            TI = TI.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag:
            TI = TI.resize((192,256), Image.BICUBIC)
        TFG_np = np.array(TI)        

        TParsing = Image.open(tparsing_path)
        TParsing_size = TParsing.size

        if TParsing_size != (1920,1080) and TParsing_size != (192,256) and BigSizeFlag:
            TParsing = TParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag and TParsing_size != (192,256):
            TParsing = TParsing.resize((192,256), Image.NEAREST)
        TParsing_np = np.array(TParsing)

        TParsing_new_np = np.zeros_like(TParsing_np)

        TParsing_new_np[(TParsing_np == 3) | (TParsing_np == 5) | (TParsing_np == 6) | (TParsing_np == 7) | (TParsing_np == 11)] = 1
        TParsing_new_np[(TParsing_np == 8) | (TParsing_np == 9) | (TParsing_np == 12)] = 2

        TParsing_new = Image.fromarray(TParsing_new_np)
        if TParsing_size != (192,256) and BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        TParsing_scaled = transform_scale(TParsing_new)*255.0

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        TFG_np[:,:,0][(TParsing_new_np == 0)] = 0
        TFG_np[:,:,1][(TParsing_new_np == 0)] = 0
        TFG_np[:,:,2][(TParsing_new_np == 0)] = 0
        TFG_scaled = transform_scale(Image.fromarray(TFG_np))

        return TParsing_scaled, TFG_scaled

    def crop(self, Ai):
        w = Ai.size()[2]
        base = 32
        x_cen = w // 2
        bs = int(w * 0.25) // base * base
        return Ai[:,:,(x_cen-bs):(x_cen+bs)]

    def __len__(self):        
        return sum(self.frames_count)

    def name(self):
        return 'ClothDataset'