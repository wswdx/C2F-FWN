import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_transform_fixed, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

#dataset for full testing of all the stages
class FullDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 

        self.dir_tpose = os.path.join(opt.dataroot, opt.phase + '_pose/target')
        self.dir_tparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/target')
        self.dir_timg = os.path.join(opt.dataroot, opt.phase + '_img/target')
        if not os.path.exists(self.dir_timg):
            self.dir_timg = os.path.join(opt.dataroot, opt.phase + '_fg/target')
        self.dir_spose = os.path.join(opt.dataroot, opt.phase + '_pose/source')
        self.dir_sparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/source')
        self.dir_simg = os.path.join(opt.dataroot, opt.phase + '_img/source')
        if not os.path.exists(self.dir_simg):
            self.dir_simg = os.path.join(opt.dataroot, opt.phase + '_fg/source')
        self.dir_bg = os.path.join(opt.dataroot, opt.phase + '_bg')

        self.tpose_paths = sorted(make_grouped_dataset(self.dir_tpose))
        self.tparsing_paths = sorted(make_grouped_dataset(self.dir_tparsing))
        self.timg_paths = sorted(make_grouped_dataset(self.dir_timg))
        self.spose_paths = sorted(make_grouped_dataset(self.dir_spose))
        self.sparsing_paths = sorted(make_grouped_dataset(self.dir_sparsing))
        self.simg_paths = sorted(make_grouped_dataset(self.dir_simg))

        self.init_frame_idx_full(self.simg_paths)

    def __getitem__(self, index):
        TPose, TParsing, TFG, TPose_uncloth, TParsing_uncloth, TFG_uncloth, TPose_cloth, TParsing_cloth, TFG_cloth, SPose, SParsing, SI, BG, BG_flag, seq_idx = self.update_frame_idx_full(self.simg_paths, index)
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

        tpose_path = self.tpose_paths[seq_idx][0]
        tparsing_path = self.tparsing_paths[seq_idx][0]
        timg_path = self.timg_paths[seq_idx][0]
        video_name = timg_path[timg_path.rfind('video'):timg_path.rfind('/timg')]
        bg_path = self.dir_bg + '/' + video_name + '.jpg'
        BG_i, BG_flag = self.get_bg_image(bg_path, size, params, BigSizeFlag)

        TPose, TParsing, TFG, TPose_uncloth, TParsing_uncloth, TFG_uncloth, TPose_cloth, TParsing_cloth, TFG_cloth = self.get_TImage(tpose_path, tparsing_path, timg_path, size, params, BigSizeFlag)
        TPose, TParsing, TFG, TPose_uncloth, TParsing_uncloth, TFG_uncloth, TPose_cloth, TParsing_cloth, TFG_cloth = self.crop(TPose), self.crop(TParsing), self.crop(TFG), self.crop(TPose_uncloth), self.crop(TParsing_uncloth), self.crop(TFG_uncloth), self.crop(TPose_cloth), self.crop(TParsing_cloth), self.crop(TFG_cloth)

        frame_range = list(range(n_frames_total+2)) if (self.opt.isTrain or self.TPose is None) else [self.opt.n_frames_G-1+2]
        for i in frame_range:
            simg_path = simg_paths[start_idx + i * t_step]
            spose_path = self.spose_paths[seq_idx][start_idx + i * t_step]
            sparsing_path = self.sparsing_paths[seq_idx][start_idx + i * t_step]

            SPose_i, SParsing_i, SI_i = self.get_SImage(spose_path, simg_path, sparsing_path, size, params, BigSizeFlag, BG_flag)

            SI_i = self.crop(SI_i)
            SPose_i = self.crop(SPose_i)
            SParsing_i = self.crop(SParsing_i)

            SPose = concat_frame(SPose, SPose_i, n_frames_total+2)
            SParsing = concat_frame(SParsing, SParsing_i, n_frames_total+2)
            SI = concat_frame(SI, SI_i, n_frames_total+2)
            BG = concat_frame(BG, BG_i, n_frames_total+2)

        if not self.opt.isTrain:
            self.TPose, self.TParsing, self.TFG, self.TPose_uncloth, self.TParsing_uncloth, self.TFG_uncloth, self.TPose_cloth, self.TParsing_cloth, self.TFG_cloth, self.SPose, self.SParsing, self.SI, self.BG, self.BG_flag = TPose, TParsing, TFG, TPose_uncloth, TParsing_uncloth, TFG_uncloth, TPose_cloth, TParsing_cloth, TFG_cloth, SPose, SParsing, SI, BG, BG_flag
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'TPose': TPose, 'TParsing': TParsing, 'TFG': TFG, 'TPose_uncloth': TPose_uncloth, 'TParsing_uncloth': TParsing_uncloth, 'TFG_uncloth': TFG_uncloth, 'TPose_cloth': TPose_cloth, 'TParsing_cloth': TParsing_cloth, 'TFG_cloth': TFG_cloth, 'SPose': SPose, 'SParsing': SParsing, 'SI': SI, 'BG': BG, 'BG_flag': BG_flag, 'A_path': simg_path, 'change_seq': change_seq}
        return return_list

    def get_bg_image(self, bg_path, size, params, BigSizeFlag):
        if os.path.exists(bg_path):
            BG = Image.open(bg_path).convert('RGB')
            if BigSizeFlag:
                transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC)
            else:
                transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC)
            BG_scaled = transform_scale(BG)
            if BigSizeFlag:
                BG_scaled = self.crop(BG_scaled)
            BG_flag = True
        else:
            print('No available background input found in: ' + bg_path)
            BG_scaled = -torch.ones(3, 256, 192)
            BG_flag = False
        return BG_scaled, BG_flag

    def get_SImage(self, spose_path, simg_path, sparsing_path, size, params, BigSizeFlag, BG_flag):          
        SI = Image.open(simg_path).convert('RGB')
        if SI.size != (1920,1080) and SI.size != (192,256) and BigSizeFlag:
            SI = SI.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag and SI.size != (192,256):
            SI = SI.resize((192,256), Image.BICUBIC)

        SParsing = Image.open(sparsing_path)
        if SParsing.size != (1920,1080) and SParsing.size != (192,256) and BigSizeFlag:
            SParsing = SParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag and SParsing.size != (192,256):
            SParsing = SParsing.resize((192,256), Image.NEAREST)
        SParsing_np = np.array(SParsing)

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
        
        SParsing_new = Image.fromarray(SParsing_new_np)
        
        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST)
        SParsing_scaled = transform_scale(SParsing_new)*255.0

        if not BG_flag:
            SI_np = np.array(SI)
            SI_np[:,:,0][SParsing_np==0] = 0
            SI_np[:,:,1][SParsing_np==0] = 0
            SI_np[:,:,2][SParsing_np==0] = 0
            SI = Image.fromarray(SI_np)

        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        SPose_array, _ = read_keypoints(spose_path, size, random_drop_prob, self.opt.remove_face_labels, self.opt.basic_point_only)
        SPose = Image.fromarray(SPose_array)         
        if SPose.size != (1920,1080) and BigSizeFlag:
            SPose = SPose.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            SPose = SPose.resize((192,256), Image.NEAREST)

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST)
        SPose_scaled = transform_scale(SPose)

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC)
        SI_scaled = transform_scale(SI)

        return SPose_scaled, SParsing_scaled, SI_scaled

    def get_TImage(self, tpose_path, tparsing_path, timg_path, size, params, BigSizeFlag):          
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        TPose_array, translation = read_keypoints(tpose_path, size, random_drop_prob, self.opt.remove_face_labels, self.opt.basic_point_only)
        TPose = Image.fromarray(TPose_array)
        if TPose.size != (1920,1080) and BigSizeFlag:
            TPose = TPose.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            TPose = TPose.resize((192,256), Image.NEAREST)
        TPose_uncloth_tmp_np = np.array(TPose)
        TPose_cloth_tmp_np = np.array(TPose)

        TI = Image.open(timg_path).convert('RGB')
        if TI.size != (1920,1080) and BigSizeFlag:
            TI = TI.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag:
            TI = TI.resize((192,256), Image.BICUBIC)
        TFG_tmp_np = np.array(TI)
        TFG_uncloth_tmp_np = np.array(TI)
        TFG_cloth_tmp_np = np.array(TI)

        TParsing = Image.open(tparsing_path)
        TParsing_size = TParsing.size
        if TParsing_size != (1920,1080) and TParsing_size != (192,256) and BigSizeFlag:
            TParsing = TParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag and TParsing_size != (192,256):
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

        TParsing_uncloth_np = np.zeros_like(TParsing_np)
        TParsing_uncloth_np[(TParsing_np == 1) | (TParsing_np == 2)] = 1
        TParsing_uncloth_np[(TParsing_np == 4) | (TParsing_np == 13)] = 2
        TParsing_uncloth_np[(TParsing_np == 14)] = 3
        TParsing_uncloth_np[(TParsing_np == 15)] = 4
        TParsing_uncloth_np[(TParsing_np == 16)] = 5
        TParsing_uncloth_np[(TParsing_np == 17)] = 6
        TParsing_uncloth_np[(TParsing_np == 10)] = 7
        TParsing_uncloth_np[(TParsing_np == 18)] = 8
        TParsing_uncloth_np[(TParsing_np == 19)] = 9

        TParsing_cloth_np = np.zeros_like(TParsing_new_np)
        TParsing_cloth_np[(TParsing_new_np == 2) | (TParsing_new_np == 3)] = \
        TParsing_new_np[(TParsing_new_np == 2) | (TParsing_new_np == 3)] - 1
        
        TParsing_new = Image.fromarray(TParsing_new_np)
        TParsing_uncloth = Image.fromarray(TParsing_uncloth_np)
        TParsing_cloth = Image.fromarray(TParsing_cloth_np)
        if TParsing_size != (192,256) and BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST)
        TParsing_uncloth_scaled = transform_scale(TParsing_uncloth)*255.0
        TParsing_cloth_scaled = transform_scale(TParsing_cloth)*255.0
        TParsing_scaled = transform_scale(TParsing_new)*255.0

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST)
        
        TPose_scaled = transform_scale(TPose)

        TPose_uncloth_tmp_np[:,:,0][(TParsing_uncloth_np == 0)] = 0
        TPose_uncloth_tmp_np[:,:,1][(TParsing_uncloth_np == 0)] = 0
        TPose_uncloth_tmp_np[:,:,2][(TParsing_uncloth_np == 0)] = 0
        TPose_uncloth_scaled = transform_scale(Image.fromarray(TPose_uncloth_tmp_np))

        TPose_cloth_tmp_np[:,:,0][(TParsing_cloth_np == 0)] = 0
        TPose_cloth_tmp_np[:,:,1][(TParsing_cloth_np == 0)] = 0
        TPose_cloth_tmp_np[:,:,2][(TParsing_cloth_np == 0)] = 0
        TPose_cloth_scaled = transform_scale(Image.fromarray(TPose_cloth_tmp_np))

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC)
        
        TFG_tmp_np[:,:,0][(TParsing_new_np == 0)] = 0
        TFG_tmp_np[:,:,1][(TParsing_new_np == 0)] = 0
        TFG_tmp_np[:,:,2][(TParsing_new_np == 0)] = 0
        TFG_scaled = transform_scale(Image.fromarray(TFG_tmp_np))

        TFG_uncloth_tmp_np[:,:,0][(TParsing_uncloth_np == 0)] = 0
        TFG_uncloth_tmp_np[:,:,1][(TParsing_uncloth_np == 0)] = 0
        TFG_uncloth_tmp_np[:,:,2][(TParsing_uncloth_np == 0)] = 0
        TFG_uncloth_scaled = transform_scale(Image.fromarray(TFG_uncloth_tmp_np))

        TFG_cloth_tmp_np[:,:,0][(TParsing_cloth_np == 0)] = 0
        TFG_cloth_tmp_np[:,:,1][(TParsing_cloth_np == 0)] = 0
        TFG_cloth_tmp_np[:,:,2][(TParsing_cloth_np == 0)] = 0
        TFG_cloth_scaled = transform_scale(Image.fromarray(TFG_cloth_tmp_np))

        return TPose_scaled, TParsing_scaled, TFG_scaled, TPose_uncloth_scaled, TParsing_uncloth_scaled, TFG_uncloth_scaled, TPose_cloth_scaled, TParsing_cloth_scaled, TFG_cloth_scaled

    def crop(self, Ai):
        w = Ai.size()[2]
        base = 32
        x_cen = w // 2
        bs = int(w * 0.25) // base * base
        return Ai[:,:,(x_cen-bs):(x_cen+bs)]

    def __len__(self):        
        return sum(self.frames_count)

    def name(self):
        return 'FullDataset'