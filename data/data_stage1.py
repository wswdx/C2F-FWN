import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_transform_fixed, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

# dataset for the Layout GAN of stage 1
class ParserDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 

        self.dir_tparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/target')
        self.dir_spose = os.path.join(opt.dataroot, opt.phase + '_pose/source')
        self.dir_sparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/source')
        self.dir_sfg = os.path.join(opt.dataroot, opt.phase + '_fg/source')

        self.tparsing_paths = sorted(make_grouped_dataset(self.dir_tparsing))
        self.spose_paths = sorted(make_grouped_dataset(self.dir_spose))
        self.sparsing_paths = sorted(make_grouped_dataset(self.dir_sparsing))
        self.sfg_paths = sorted(make_grouped_dataset(self.dir_sfg))

        self.init_frame_idx_parser(self.sparsing_paths)

    def __getitem__(self, index):
        TParsing, SPose, SParsing, SFG, seq_idx = self.update_frame_idx_parser(self.sparsing_paths, index)
        sparsing_paths = self.sparsing_paths[seq_idx]        
        sfg_paths = self.sfg_paths[seq_idx]
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(sparsing_paths), self.frame_idx)
        
        sparsing = Image.open(sparsing_paths[start_idx])
        sfg = Image.open(sfg_paths[start_idx]).convert('RGB')
        size = sfg.size

        BigSizeFlag = True
        if size[0]/size[1] > 1:
            BigSizeFlag = True
        else:
            BigSizeFlag = False

        if BigSizeFlag:
            params = get_img_params(self.opt, (1920, 1080))
        else:
            params = get_img_params(self.opt, size)

        tparsing_path = self.tparsing_paths[seq_idx][0]

        TParsing = self.get_TImage(tparsing_path, size, params, BigSizeFlag)
        TParsing = self.crop(TParsing)

        frame_range = list(range(n_frames_total)) if (self.opt.isTrain or self.TPose is None) else [self.opt.n_frames_G-1]
        for i in frame_range:
            sparsing_path = sparsing_paths[start_idx + i * t_step]
            spose_path = self.spose_paths[seq_idx][start_idx + i * t_step]
            sfg_path = sfg_paths[start_idx + i * t_step]

            SPose_i, SParsing_i, SFG_i = self.get_SImage(spose_path, sparsing_path, sfg_path, size, params, BigSizeFlag)

            SParsing_i = self.crop(SParsing_i)
            SPose_i = self.crop(SPose_i)
            SFG_i = self.crop(SFG_i)

            SPose = concat_frame(SPose, SPose_i, n_frames_total)
            SParsing = concat_frame(SParsing, SParsing_i, n_frames_total)
            SFG = concat_frame(SFG, SFG_i, n_frames_total)

        if not self.opt.isTrain:
            self.TParsing, self.SPose, self.SParsing, self.SFG = TParsing, SPose, SParsing, SFG
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'TParsing': TParsing, 'SPose': SPose, 'SParsing': SParsing, 'SFG': SFG, 'A_path': sparsing_path, 'change_seq': change_seq}
        return return_list

    def get_SImage(self, spose_path, sparsing_path, sfg_path, size, params, BigSizeFlag):          
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        SFG = Image.open(sfg_path).convert('RGB')
        SPose_array, _ = read_keypoints(spose_path, size, random_drop_prob, self.opt.remove_face_labels, self.opt.basic_point_only)
        SPose = Image.fromarray(SPose_array)         
        if SPose.size != (1920,1080) and BigSizeFlag:
            SPose = SPose.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            SPose = SPose.resize((192,256), Image.NEAREST)
        SPose_np = np.array(SPose)

        SParsing = Image.open(sparsing_path)
        SParsing_size = SParsing.size
        if SParsing_size != (1920,1080) and BigSizeFlag:
            SParsing = SParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            SParsing = SParsing.resize((192,256), Image.NEAREST)
        SParsing_np = np.array(SParsing)

        if SFG.size != (1920,1080) and BigSizeFlag:
            SFG = SFG.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag and SFG.size != (192,256):
            SFG = SFG.resize((192,256), Image.BICUBIC)

        SParsing_new_np = np.zeros_like(SParsing_np)
        SParsing_new_np[(SParsing_np == 1) | (SParsing_np == 2)] = 1
        SParsing_new_np[(SParsing_np == 3) | (SParsing_np == 5) | (SParsing_np == 6) | (SParsing_np == 7) | (SParsing_np == 11)] = 2
        SParsing_new_np[(SParsing_np == 8) | (SParsing_np == 9) | (SParsing_np == 12)] = 3
        SParsing_new_np[(SParsing_np == 4) | (SParsing_np == 13)] = 4
        SParsing_new_np[(SParsing_np == 14)] = 5
        SParsing_new_np[(SParsing_np == 15)] = 6
        SParsing_new_np[(SParsing_np == 16)] = 7
        SParsing_new_np[(SParsing_np == 17)] = 8
        SParsing_new_np[(SParsing_np == 10)] = 9
        SParsing_new_np[(SParsing_np == 18)] = 10
        SParsing_new_np[(SParsing_np == 19)] = 11

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST)

        SPose_scaled = transform_scale(Image.fromarray(SPose_np))

        SParsing_new = Image.fromarray(SParsing_new_np)
        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST)
        SParsing_scaled = transform_scale(SParsing_new)*255.0

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC)
        SFG_scaled = transform_scale(SFG)

        return SPose_scaled, SParsing_scaled, SFG_scaled

    def get_TImage(self, tparsing_path, size, params, BigSizeFlag):          
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        TParsing = Image.open(tparsing_path)
        TParsing_size = TParsing.size
        if TParsing_size != (1920,1080) and BigSizeFlag:
            TParsing = TParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            TParsing = TParsing.resize((192,256), Image.NEAREST)
        TParsing_np = np.array(TParsing)

        TParsing_new_np = np.zeros_like(TParsing_np)
        TParsing_new_np[(TParsing_np == 1) | (TParsing_np == 2)] = 1
        TParsing_new_np[(TParsing_np == 3) | (TParsing_np == 5) | (TParsing_np == 6) | (TParsing_np == 7) | (TParsing_np == 11)] = 2
        TParsing_new_np[(TParsing_np == 8) | (TParsing_np == 9) | (TParsing_np == 12)] = 3
        TParsing_new_np[(TParsing_np == 4) | (TParsing_np == 13)] = 4
        TParsing_new_np[(TParsing_np == 14)] = 5
        TParsing_new_np[(TParsing_np == 15)] = 6
        TParsing_new_np[(TParsing_np == 16)] = 7
        TParsing_new_np[(TParsing_np == 17)] = 8
        TParsing_new_np[(TParsing_np == 10)] = 9
        TParsing_new_np[(TParsing_np == 18)] = 10
        TParsing_new_np[(TParsing_np == 19)] = 11

        TParsing_new = Image.fromarray(TParsing_new_np)
        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST)
        TParsing_scaled = transform_scale(TParsing_new)*255.0

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST)

        return TParsing_scaled

    def crop(self, Ai):
        w = Ai.size()[2]
        base = 32
        x_cen = w // 2
        bs = int(w * 0.25) // base * base
        return Ai[:,:,(x_cen-bs):(x_cen+bs)]

    def __len__(self):        
        return sum(self.frames_count)

    def name(self):
        return 'ParserDataset'