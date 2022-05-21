"""
Evaluate transition model
"""

import os
import random
import numpy as np
import os.path as osp
from tqdm import tqdm
import joblib

import torch
from models.transmodel import ParticleNet
from datasets.dataset_splishsplash_rawdata import ParticleDataset
from utils.particles_utils import record2obj
from utils.point_eval import FluidErrors


class TransModelEvaluation():
    def __init__(self, options):
        self.seed_everything(10)
        self.options = options
        self.device = torch.device('cuda')

        self.exppath = osp.join(self.options.expdir, self.options.expname)
        
        gravity = self.options.TEST.gravity
        self.transition_model = ParticleNet(gravity=gravity).to(self.device)
        ckpt = torch.load(self.options.resume_from)
        if 'transition_model_state_dict' in ckpt:
            ckpt = ckpt['transition_model_state_dict']
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        ckpt = {k:v for k,v in ckpt.items() if 'gravity' not in k}
        transition_model_state_dict = self.transition_model.state_dict()
        transition_model_state_dict.update(ckpt)
        self.transition_model.load_state_dict(transition_model_state_dict, strict=True)
        
        self.dataset = ParticleDataset(data_path=self.options.TEST.datapath, 
                                       data_type=self.options.TEST.datatype,
                                       start=self.options.TEST.start_index,
                                       end=self.options.TEST.end_index,
                                       random_rot=False, window=2)
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
                    self.pos_for_next_step, self.vel_for_next_step = data['particles_pos_0'],data['particles_vel_0']
                pred_pos, pred_vel, num_fluid_nn = self.transition_model(self.pos_for_next_step, self.vel_for_next_step, box, box_normals)
                self.pos_for_next_step, self.vel_for_next_step = pred_pos.clone().detach(),pred_vel.clone().detach()

                # calculate pred2gt distance
                dist_pred2gt = self.fluid_erros.cal_errors(pred_pos.cpu().numpy(), gt_pos.cpu().numpy(), data_idx+1)
                dist_pred2gt_all.append(dist_pred2gt)
                
                # calculate pred2gt distance
                cliped_dist_pred2gt = self.cliped_fluid_erros.cal_errors(self.strict_clip_particles(pred_pos).cpu().numpy(), self.strict_clip_particles(gt_pos).cpu().numpy(), data_idx+1)
                cliped_dist_pred2gt_all.append(cliped_dist_pred2gt)
                

                if not os.path.exists(osp.join(self.exppath, 'clip')):
                    os.makedirs(osp.join(self.exppath, 'clip'))
                
                if self.options.TEST.save_obj:
                    particle_name = osp.join(self.exppath, f'pred_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(pred_pos, fp, color=[255, 0, 0]) # red
                    particle_name = osp.join(self.exppath, f'gt_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(gt_pos, fp, color=[3, 168, 158])
                    
                    # cliped
                    particle_name = osp.join(self.exppath, 'clip', f'pred_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(self.strict_clip_particles(pred_pos), fp, color=[255, 0, 0]) # red
                    particle_name = osp.join(self.exppath, 'clip', f'gt_{data_idx+1}.obj')
                    with open(particle_name, 'w') as fp:
                        record2obj(self.strict_clip_particles(gt_pos), fp, color=[3, 168, 158])
                        
            self.fluid_erros.save(osp.join(self.exppath, 'res.json'))
            self.cliped_fluid_erros.save(osp.join(self.exppath, 'clip', 'res.json'))
        print('\n----------------- trained 50 steps ------------------------')
        print('Pred2GT:', np.mean(dist_pred2gt_all[0:49]))
        print('Pred2GT-10:', np.mean(dist_pred2gt_all[:10]))
        print('Pred2GT-end:', dist_pred2gt_all[48])
        
        print('\n----------------- rollout 10 steps ------------------------')
        print('Pred2GT:', np.mean(dist_pred2gt_all[-10:]))
        print('Pred2GT-5:', np.mean(dist_pred2gt_all[-5]))
        print('Pred2GT-end:', dist_pred2gt_all[-1])
        # save
        joblib.dump({'pred2gt': dist_pred2gt_all, 'cham_dist_all': cham_dist_all}, os.path.join(self.exppath, 'res.pt'))
        
        # ---> clip
        print('\n----------------- clipped trained 50 steps ------------------------')
        print('Pred2GT:', np.mean(cliped_dist_pred2gt_all[:49]))
        print('Pred2GT-10:', np.mean(cliped_dist_pred2gt_all[:10]))
        print('Pred2GT-end:', cliped_dist_pred2gt_all[48])
        
        print('\n----------------- rollout 10 steps ------------------------')
        print('Pred2GT:', np.mean(cliped_dist_pred2gt_all[-10:]))
        print('Pred2GT-5:', np.mean(cliped_dist_pred2gt_all[-5:]))
        print('Pred2GT-end:', cliped_dist_pred2gt_all[-1])
        # save
        joblib.dump({'pred2gt': cliped_dist_pred2gt_all, 'cham_dist_all': cliped_cham_dist_all}, os.path.join(self.exppath, 'clip', 'res.pt'))


if __name__ == '__main__':
    from configs import transmodel_config

    cfg = transmodel_config()
    evaluator = TransModelEvaluation(cfg)
    evaluator.eval()
