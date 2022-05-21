"""
Evaluate transition model
"""

import os
import random
import imageio
import numpy as np
import os.path as osp
from tqdm import tqdm
from time import time
import joblib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.particleModel import ParticleNet
from datasets.dataset import BlenderDataset
from utils.vis_particles import record2obj
from utils.point_eval import FluidErrors
from pytorch3d.loss import chamfer_distance


class TransModelEvaluation():
    def __init__(self, options):
        self.seed_everything(10)
        self.options = options
        print(self.options.start_index, self.options.end_index)
        self.exppath = osp.join(options.expdir, options.expname)
        os.makedirs(self.exppath, exist_ok=True)
        self.device = torch.device('cuda')
        
        self.transition_model = ParticleNet(gravity=(0,0, -9.81)).to(self.device)
        ckpt = torch.load(self.options.pretrained_ckpt)
        if 'transition_model_state_dict' in ckpt:
            ckpt = ckpt['transition_model_state_dict']
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        ckpt = {k:v for k,v in ckpt.items() if 'gravity' not in k}
        transition_model_state_dict = self.transition_model.state_dict()
        transition_model_state_dict.update(ckpt)
        self.transition_model.load_state_dict(transition_model_state_dict, strict=True)
        
        self.dataset = BlenderDataset(self.options.data_path, self.options, split='test')
        self.dataset_length = len(self.dataset)
        
        self.fluid_erros = FluidErrors()
        self.cliped_fluid_erros = FluidErrors()
        self.init_box_boundary()
        
    def init_box_boundary(self):
        particle_radius = 0.025
        self.x_bound = [1-particle_radius, -1+particle_radius]
        self.y_bound = [1-particle_radius, -1+particle_radius]
        self.z_bound = [2.4552-particle_radius, -1+particle_radius]
    
    def strict_clip_particles(self, pos):
        assert len(pos.shape) == 2
        clipped_x = torch.clamp(pos[:, 0], max=self.x_bound[0], min=self.x_bound[1])
        clipped_y = torch.clamp(pos[:, 1], max=self.y_bound[0], min=self.y_bound[1])
        clipped_z = torch.clamp(pos[:, 2], max=self.z_bound[0], min=self.z_bound[1])
        clipped_pos = torch.stack((clipped_x, clipped_y, clipped_z), dim=1)
        return clipped_pos
        
    def seed_everything(self, seed):
        """
        ensure reproduction
        """
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        print('---> seed has been set')
        

    def eval(self, save_obj=False):
        print(self.options.expname)
        # self.transition_model.eval()
        dist_pred2gt_all = []
        vel_err_all = []
        cham_dist_all = []
        cliped_dist_pred2gt_all = []
        cliped_cham_dist_all = []
        with torch.no_grad():
            for data_idx in tqdm(range(self.dataset_length), total=self.dataset_length, desc='Eval:'):
                data = self.dataset[data_idx]
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}

                box = data['box']
                box_normals = data['box_normals']
                gt_pos = data['particles_pos_1']
                gt_vel = data['particles_vel_1']
                if data_idx == 0:
                    self.pos_for_next_step, self.vel_for_next_step = data['particles_pos'],data['particles_vel']
                pred_pos, pred_vel, num_fluid_nn = self.transition_model(self.pos_for_next_step, self.vel_for_next_step, box, box_normals)
                self.pos_for_next_step, self.vel_for_next_step = pred_pos.clone().detach(),pred_vel.clone().detach()

                # calculate pred2gt distance
                dist_pred2gt = self.fluid_erros.cal_errors(pred_pos.cpu().numpy(), gt_pos.cpu().numpy(), data_idx+1)
                dist_pred2gt_all.append(dist_pred2gt)
                
                # # calculate vel error
                # vel_err = F.l1_loss(pred_vel, gt_vel)
                # vel_err_all.append(vel_err.item())
                
                # calcuulate chamfer dist
                cham_dist,_ = chamfer_distance(pred_pos.unsqueeze(0), gt_pos.unsqueeze(0))
                cham_dist_all.append(cham_dist.item())
                
                # ---> clip particle pos
                # calculate pred2gt distance
                cliped_dist_pred2gt = self.cliped_fluid_erros.cal_errors(self.strict_clip_particles(pred_pos).cpu().numpy(), self.strict_clip_particles(gt_pos).cpu().numpy(), data_idx+1)
                cliped_dist_pred2gt_all.append(cliped_dist_pred2gt)
                
                # # calculate vel error
                # cliped_vel_err = F.l1_loss(pred_vel, gt_vel)
                # cliped_vel_err_all.append(vel_err.item())
                
                # calcuulate chamfer dist
                cliped_cham_dist,_ = chamfer_distance(self.strict_clip_particles(pred_pos).unsqueeze(0), self.strict_clip_particles(gt_pos).unsqueeze(0))
                cliped_cham_dist_all.append(cliped_cham_dist.item())
                # ---> clip particle pos

                if not os.path.exists(osp.join(self.exppath, 'clip')):
                    os.makedirs(osp.join(self.exppath, 'clip'))
                
                if save_obj:
                    # save obj
                    particle_name = osp.join(self.exppath, f'pred_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(self.strict_clip_particles(pred_pos), fp, color=[255, 0, 0]) # red
                    particle_name = osp.join(self.exppath, f'gt_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(self.strict_clip_particles(gt_pos), fp, color=[3, 168, 158])
                    
                    # save obj (cliped)
                    particle_name = osp.join(self.exppath, 'clip', f'pred_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(self.strict_clip_particles(pred_pos), fp, color=[255, 0, 0]) # red
                    particle_name = osp.join(self.exppath, 'clip', f'gt_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(self.strict_clip_particles(gt_pos), fp, color=[3, 168, 158])
                        
            self.fluid_erros.save(osp.join(self.exppath, 'res.json'))
            if not os.path.exists(osp.join(self.exppath, 'clip')):
                os.makedirs(osp.join(self.exppath, 'clip'))
            self.cliped_fluid_erros.save(osp.join(self.exppath, 'clip', 'res.json'))
        # print(len(dist_pred2gt_all))
        # exit()
        print('Pred2GT:', np.mean(dist_pred2gt_all[0:49]))
        # print('Pred2GT-10:', np.mean(dist_pred2gt_all[:10]))
        print('Pred2GT-end:', dist_pred2gt_all[48])
        # print('ChanDist:',  np.mean(cham_dist_all[:50])*1000)
        # print('ChanDist-10:',  np.mean(cham_dist_all[:10])*1000)
        
        print('-----------------rollout 10 steps------------------------')
        print('Pred2GT:', np.mean(dist_pred2gt_all[49:59]))
        print('Pred2GT-5:', np.mean(dist_pred2gt_all[49:54]))
        print('Pred2GT-end:', dist_pred2gt_all[58])
        # print('ChanDist:',  np.mean(cham_dist_all[50:60])*1000)
        # print('ChanDist-5:',  np.mean(cham_dist_all[50:55])*1000)
        # print('Vel Err:', np.mean(vel_err_all))
        with open(os.path.join(self.exppath, f'{np.mean(dist_pred2gt_all)}.txt'), 'w') as fp:
            fp.write(f'{np.mean(dist_pred2gt_all)}\n')
        # save
        joblib.dump({'pred2gt': dist_pred2gt_all, 'cham_dist_all': cham_dist_all}, os.path.join(self.exppath, 'res.pt'))
        # joblib.dump({'vel_errs':vel_err_all, 'pred2gt': dist_pred2gt_all, 'cham_dist_all': cham_dist_all}, os.path.join(self.exppath, 'res.pt'))
        
        # ---> clip
        # print('\n-----------------clipped result------------------------')
        # print('Pred2GT:', np.mean(cliped_dist_pred2gt_all[:50]))
        # print('Pred2GT-10:', np.mean(cliped_dist_pred2gt_all[:10]))
        # print('Pred2GT-end:', cliped_dist_pred2gt_all[49])
        # print('ChanDist:',  np.mean(cliped_cham_dist_all[:50])*1000)
        # print('ChanDist-10:',  np.mean(cliped_cham_dist_all[:10])*1000)
        
        # print('-----------------rollout 10 steps------------------------')
        # print('Pred2GT:', np.mean(cliped_dist_pred2gt_all[50:60]))
        # print('Pred2GT-5:', np.mean(cliped_dist_pred2gt_all[50:55]))
        # print('Pred2GT-end:', cliped_dist_pred2gt_all[59])
        # print('ChanDist:',  np.mean(cliped_cham_dist_all[50:60])*1000)
        # print('ChanDist-5:',  np.mean(cliped_cham_dist_all[50:55])*1000)
        # print('Vel Err:', np.mean(vel_err_all))
        with open(os.path.join(self.exppath, f'{np.mean(dist_pred2gt_all)}.txt'), 'w') as fp:
            fp.write(f'{np.mean(cliped_dist_pred2gt_all)}\n')
        # save
        joblib.dump({'pred2gt': cliped_dist_pred2gt_all, 'cham_dist_all': cliped_cham_dist_all}, os.path.join(self.exppath, 'clip', 'res.pt'))
        # joblib.dump({'vel_errs':vel_err_all, 'pred2gt': dist_pred2gt_all, 'cham_dist_all': cham_dist_all}, os.path.join(self.exppath, 'res.pt'))
        # ---> clip