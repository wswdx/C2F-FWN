import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_transform_fixed, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

# dataset for the Composition GAN of stage 3
class ComposerDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 

        self.dir_tparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/target')
        self.dir_timg = os.path.join(opt.dataroot, opt.phase + '_img/target')
        self.dir_spose = os.path.join(opt.dataroot, opt.phase + '_pose/source')
        self.dir_sparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/source')
        self.dir_sfg = os.path.join(opt.dataroot, opt.phase + '_fg/source')
        self.dir_simg = os.path.join(opt.dataroot, opt.phase + '_img/source')
        self.dir_bg = os.path.join(opt.dataroot, opt.phase + '_bg')
             
        self.tparsing_paths = sorted(make_grouped_dataset(self.dir_tparsing))
        self.timg_paths = sorted(make_grouped_dataset(self.dir_timg))
        self.spose_paths = sorted(make_grouped_dataset(self.dir_spose))
        self.sparsing_paths = sorted(make_grouped_dataset(self.dir_sparsing))
        self.sfg_paths = sorted(make_grouped_dataset(self.dir_sfg))
        self.simg_paths = sorted(make_grouped_dataset(self.dir_simg))

        self.init_frame_idx_composer(self.simg_paths)

    def __getitem__(self, index):
        TParsing, TFG, SPose, SParsing, SFG, SFG_full, BG, BG_flag, SI, seq_idx = self.update_frame_idx_composer(self.simg_paths, index)
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

        video_name = timg_path[timg_path.rfind('video'):timg_path.rfind('/timg')]
        bg_path = self.dir_bg + '/' + video_name + '.jpg'
        BG_i, BG_flag = self.get_bg_image(bg_path, size, params, BigSizeFlag)

        TParsing, TFG = self.get_TImage(tparsing_path, timg_path, size, params, BigSizeFlag)
        TParsing, TFG = self.crop(TParsing), self.crop(TFG)

        frame_range = list(range(n_frames_total)) if (self.opt.isTrain or self.TPose is None) else [self.opt.n_frames_G-1]
        for i in frame_range:
            simg_path = simg_paths[start_idx + i * t_step]
            sfg_path = self.sfg_paths[seq_idx][start_idx + i * t_step]
            spose_path = self.spose_paths[seq_idx][start_idx + i * t_step]
            sparsing_path = self.sparsing_paths[seq_idx][start_idx + i * t_step]

            SPose_i, SParsing_i, SFG_i, SFG_full_i, SI_i = self.get_SImage(spose_path, sparsing_path, sfg_path, simg_path, size, params, BigSizeFlag)

            SParsing_i = self.crop(SParsing_i)
            SFG_i = self.crop(SFG_i)
            SPose_i, SFG_full_i, SI_i = self.crop(SPose_i), self.crop(SFG_full_i), self.crop(SI_i)

            SPose = concat_frame(SPose, SPose_i, n_frames_total)
            SParsing = concat_frame(SParsing, SParsing_i, n_frames_total)
            SFG = concat_frame(SFG, SFG_i, n_frames_total)
            SFG_full = concat_frame(SFG_full, SFG_full_i, n_frames_total)
            SI = concat_frame(SI, SI_i, n_frames_total)
            BG = concat_frame(BG, BG_i, n_frames_total)

        if not self.opt.isTrain:
            self.TParsing, self.TFG, self.SPose, self.SParsing, self.SFG, self.SFG_full, self.BG, self.BG_flag, self.SI = TParsing, TFG, SPose, SParsing, SFG, SFG_full, BG, BG_flag, SI
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'TParsing': TParsing, 'TFG': TFG, 'SPose': SPose, 'SParsing': SParsing, 'SFG': SFG, 'SFG_full': SFG_full, 'BG': BG, 'BG_flag': BG_flag, 'SI': SI, 'A_path': simg_path, 'change_seq': change_seq}
        return return_list

    def get_bg_image(self, bg_path, size, params, BigSizeFlag):
        if os.path.exists(bg_path):
            BG = Image.open(bg_path).convert('RGB')
            if BigSizeFlag:
                transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
            else:
                transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
            BG_scaled = transform_scale(BG)
            BG_scaled = self.crop(BG_scaled)
            BG_flag = True
        else:
            BG_scaled = -torch.ones(3, 256, 192)
            BG_flag = False
        return BG_scaled, BG_flag

    def get_SImage(self, spose_path, sparsing_path, sfg_path, simg_path, size, params, BigSizeFlag):          
        SI = Image.open(simg_path).convert('RGB')
        if SI.size != (1920,1080) and BigSizeFlag:
            SI = SI.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag:
            SI = SI.resize((192,256), Image.BICUBIC)
        SFG_np = np.array(SI)
        SFG_full_np = np.array(SI)

        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        SPose_array, _ = read_keypoints(spose_path, size, random_drop_prob, self.opt.remove_face_labels, self.opt.basic_point_only)
        SPose = Image.fromarray(SPose_array)         
        if SPose.size != (1920,1080) and BigSizeFlag:
            SPose = SPose.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            SPose = SPose.resize((192,256), Image.NEAREST)
        SPose_np = np.array(SPose)

        SParsing = Image.open(sparsing_path)
        SParsing_size = SParsing.size
        if SParsing_size != (1920,1080) and SParsing_size != (192,256) and BigSizeFlag:
            SParsing = SParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag and SParsing_size != (192,256):
            SParsing = SParsing.resize((192,256), Image.NEAREST)
        SParsing_np = np.array(SParsing)

        if SParsing_size == (192,256):
            SParsing_new_np = SParsing_np
        else:
            SParsing_new_np = np.zeros_like(SParsing_np)
            SParsing_new_np[(SParsing_np == 3) | (SParsing_np == 5) | (SParsing_np == 6) | (SParsing_np == 7) | (SParsing_np == 11)] = 1
            SParsing_new_np[(SParsing_np == 8) | (SParsing_np == 9) | (SParsing_np == 12)] = 2
            SParsing_new_np[(SParsing_np == 1) | (SParsing_np == 2)] = 3
            SParsing_new_np[(SParsing_np == 4) | (SParsing_np == 13)] = 4
            SParsing_new_np[(SParsing_np == 14)] = 5
            SParsing_new_np[(SParsing_np == 15)] = 6
            SParsing_new_np[(SParsing_np == 16)] = 7
            SParsing_new_np[(SParsing_np == 17)] = 8
            SParsing_new_np[(SParsing_np == 10)] = 9
            SParsing_new_np[(SParsing_np == 18)] = 10
            SParsing_new_np[(SParsing_np == 19)] = 11

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)

        SPose_scaled = transform_scale(Image.fromarray(SPose_np))

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
        SI_scaled = transform_scale(SI)
        SFG_full_np[(SParsing_new_np == 0)] = 0
        SFG_full_scaled = transform_scale(Image.fromarray(SFG_full_np))

        if SI.size != (192,256) and BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        
        if SI.size != (192,256):
            SFG_np[(SParsing_new_np != 1) & (SParsing_new_np != 2) & (SParsing_new_np != 3)] = 0
        SFG_scaled = transform_scale(Image.fromarray(SFG_np))

        return SPose_scaled, SParsing_scaled, SFG_scaled, SFG_full_scaled, SI_scaled

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
        TParsing_new_np[(TParsing_np == 1) | (TParsing_np == 2)] = 1
        TParsing_new_np[(TParsing_np == 4) | (TParsing_np == 13)] = 2
        TParsing_new_np[(TParsing_np == 14)] = 3
        TParsing_new_np[(TParsing_np == 15)] = 4
        TParsing_new_np[(TParsing_np == 16)] = 5
        TParsing_new_np[(TParsing_np == 17)] = 6
        TParsing_new_np[(TParsing_np == 10)] = 7
        TParsing_new_np[(TParsing_np == 18)] = 8
        TParsing_new_np[(TParsing_np == 19)] = 9

        TParsing_new = Image.fromarray(TParsing_new_np)
        if TParsing_size != (192,256) and BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        TParsing_scaled = transform_scale(TParsing_new)*255.0

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
        return 'ComposerDataset'