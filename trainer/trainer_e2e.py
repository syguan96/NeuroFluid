"""
The whole framework
"""

import os
import numpy as np
import os.path as osp
from tqdm import tqdm
from time import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .basetrainer import BaseTrainer
from models.renderer import RenderNet
from models.transmodel import ParticleNet
from datasets.dataset import BlenderDataset
from utils.particles_utils import record2obj
from utils.point_eval import FluidErrors
from utils.lr_schedulers import ExponentialLR


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Trainer(BaseTrainer):
    def init_fn(self):
        self.start_step = 0
        self.eval_count = 0
        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.set_RGB_criterion()
        self.set_L1_criterion()
        self.save_interval = self.options.TRAIN.save_interval
        self.log_interval = self.options.TRAIN.log_interval
            

    def build_dataloader(self):
        self.train_view_names = self.options['train'].views.dynamic
        self.test_viewnames = self.options['test'].views
        self.dataset = BlenderDataset(self.options.train.path, self.options,
                                            start_index=self.options['train'].start_index, end_index=self.options['train'].end_index,
                                            imgW=self.options.TRAIN.imgW, imgH=self.options.TRAIN.imgH,
                                            imgscale=self.options.TRAIN.scale, viewnames=self.train_view_names, split='train')
        self.dataset_length = len(self.dataset)
        self.test_dataset = BlenderDataset(self.options.test.path, self.options, 
                                            start_index=self.options['test'].start_index, end_index=self.options['test'].end_index,
                                            imgW=self.options.TEST.imgW, imgH=self.options.TEST.imgH,
                                            imgscale=self.options.TEST.scale, viewnames=self.test_viewnames, split='test')
        self.test_dataset_length = len(self.test_dataset)
        print('---> dataloader has been build')
       
        
    def build_model(self):
        # build model
        gravity = self.options.gravity
        print('---> set gravity', gravity)
        self.transition_model = ParticleNet(gravity=gravity).to(self.device)
        self.renderer = RenderNet(self.options.RENDERER, near=self.options.near, far=self.options.far).to(self.device)
        
        # load pretrained checkpoints
        if self.options.TRAIN.pretrained_transition_model != '':
            self.load_pretained_transition_model(self.options.TRAIN.pretrained_transition_model)
        if self.options.TRAIN.pretained_renderer != '':
            self.load_pretained_renderer_model(self.options.TRAIN.pretained_renderer, partial_load=self.options.TRAIN.partial_load)
    
        
    def build_optimizer(self):
        renderer_lr = self.options.TRAIN.LR.renderer_lr
        transition_lr = self.options.TRAIN.LR.trans_lr
        seperate_render_transition = self.options.TRAIN.seperate_render_transition
        if seperate_render_transition:
            self.optimizer = torch.optim.Adam([
                {'params': self.renderer.parameters(), 'lr': renderer_lr}
            ])
            self.transition_optimizer = torch.optim.Adam([
                {'params': self.transition_model.parameters(), 'lr': transition_lr},
            ])
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.renderer.parameters(), 'lr': renderer_lr},
                {'params': self.transition_model.parameters(), 'lr': transition_lr}
                ])
        if self.options.TRAIN.LR.use_scheduler:
            boundaries = [
                10000,  # 10k
                75000,  # 75k
                150000,  # 150k
            ]
            lr_values = [
                1.0,
                0.5,
                0.25,
                0.125,
                # 0.5 * 0.125,
                # 0.25 * 0.125,
                # 0.125 * 0.125,
            ]

            def lrfactor_fn(x):
                factor = lr_values[0]
                for b, v in zip(boundaries, lr_values[1:]):
                    if x > b:
                        factor = v
                    else:
                        break
                return factor

            self.optim_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lrfactor_fn)
            
            if seperate_render_transition:
                boundaries_trans = [
                    10000, # 10k
                    30000, 
                    50000,
                    100000,
                    300000,
                ]
                lr_values_trans = [
                    1.0,
                    0.5,
                    0.25,
                    0.125,
                    0.5 * 0.125,
                    0.25 * 0.125,
                    0.125 * 0.125,
                ]

                def lrfactor_fn_transition(x):
                    factor = lr_values[0]
                    for b, v in zip(boundaries_trans, lr_values_trans[1:]):
                        if x > b:
                            factor = v
                        else:
                            break
                    return factor

                self.optim_lr_scheduler_transition = torch.optim.lr_scheduler.LambdaLR(self.transition_optimizer, lrfactor_fn_transition)



    def resume(self, ckpt_file):
        checkpoint = torch.load(ckpt_file)
        self.start_step = checkpoint['step']
        self.renderer.load_state_dict(checkpoint['renderer_state_dict'], strict=True)
        self.transition_model.load_state_dict(checkpoint['transition_model_state_dict'], strict=True)


    def save_checkpoint(self, global_step):
        model_dicts = {'step':global_step,
                        'renderer_state_dict':self.renderer.state_dict(),
                        'transition_model_state_dict':self.transition_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(model_dicts, 
                    osp.join(self.exppath, 'models', f'{global_step}.pt'))    
        
    
    def train(self,):
        # prepare training
        global_step = self.start_step
        view_num = len(self.train_view_names)
        imgW, imgH = self.options.TRAIN.imgW, self.options.TRAIN.imgH
        img_scale = self.options.TRAIN.scale
        H = int(imgH // img_scale)
        W = int(imgW // img_scale)

        self.transition_model.train()
        self.renderer.train()

        for epoch_idx in tqdm(range(self.start_step, self.options.TRAIN.epochs), total=self.options.TRAIN.epochs, desc='Epoch:'):
            self.tmp_fluid_error = FluidErrors()
            for data_idx in range(self.dataset_length):
                data = self.dataset[data_idx]
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}
                # training
                loss = self.train_step(data, data_idx, view_num, H, W, global_step)
                self.update_step(loss, global_step)
                global_step += 1
                        
                # evaluation
                if (global_step+1) % self.save_interval == 0:
                    self.eval(global_step)
                    self.save_checkpoint(global_step)


    def trainsition_step_for_training(self, data, data_idx):
        box = data['box']
        box_normals = data['box_normals']
        if data_idx == 0:
            self.pos_for_next_step, self.vel_for_next_step = data['particles_pos'],data['particles_vel']
        pred_pos, pred_vel, num_fluid_nn = self.transition_model(self.pos_for_next_step, self.vel_for_next_step, box, box_normals)
        
        self.pos_for_next_step, self.vel_for_next_step = pred_pos.clone().detach(),pred_vel.clone().detach()
        self.pos_for_next_step.requires_grad = False
        self.vel_for_next_step.requires_grad = False
        return pred_pos
    
        
    def train_step(self, data, data_idx, view_num, H, W, global_step):
        # -----
        # particle transition
        # -----
        pred_pos = self.trainsition_step_for_training(data, data_idx)

        if (global_step+1) % self.log_interval == 0:
            pos_t1 = data['particles_pos_1']
            dist_pred2gt = self.tmp_fluid_error.cal_errors(pred_pos.detach().cpu().numpy(), pos_t1.detach().cpu().numpy(), data_idx+1)
            self.summary_writer.add_scalar(f'Train/pred2gt_distance', dist_pred2gt, global_step)
        
        # -----
        # rendering
        # -----
        ray_chunk = self.options.RENDERER.ray.ray_chunk
        N_importance = self.options.RENDERER.ray.N_importance
        total_loss = 0.
        for view_idx in range(view_num):
            # -------
            # render by a nerf model, and then calculate mse loss
            # -------
            view_name = self.train_view_names[view_idx]
            cw_t1 = data['cw_1'][view_idx]
            rgbs_t1 = data['rgb_1'][view_idx]
            focal_length = data['focal'][view_idx]
            rays_t1 = data['rays_1'][view_idx]
            # randomly sample pixel
            coords = self.random_sample_coords(H,W,global_step)
            coords = torch.reshape(coords, [-1,2])
            select_inds = np.random.choice(coords.shape[0], size=[ray_chunk], replace=False)
            select_coords = coords[select_inds].long()
            rays_t1 = rays_t1[select_coords[:, 0], select_coords[:, 1]]
            rgbs_t1 = rgbs_t1.view(H, W, -1)[select_coords[:, 0], select_coords[:, 1]]
            ro_t1 = self.renderer.set_ro(cw_t1)            
            render_ret = self.render_image(pred_pos, ray_chunk, ro_t1, rays_t1, focal_length, cw_t1)
            # calculate mse loss
            rgbloss_0 = self.rgb_criterion(render_ret['pred_rgbs_0'], rgbs_t1[:ray_chunk])
            if N_importance>0:
                rgbloss_1 = self.rgb_criterion(render_ret['pred_rgbs_1'], rgbs_t1[:ray_chunk])
                rgbloss = rgbloss_0 + rgbloss_1
            else:
                rgbloss = rgbloss_0
            total_loss = total_loss+rgbloss
                    
            # log
            if (global_step+1) % self.log_interval == 0:
                self.summary_writer.add_scalar(f'{view_name}/rgbloss_0', rgbloss_0.item(), global_step)
                self.summary_writer.add_scalar(f'{view_name}/rgbloss', rgbloss.item(), global_step)
                self.summary_writer.add_histogram(f'{view_name}/num_neighbors_0', render_ret['num_nn_0'], global_step)
                if N_importance>0:
                    self.summary_writer.add_scalar(f'{view_name}/rgbloss_1', rgbloss_1.item(), global_step)
                    self.summary_writer.add_histogram(f'{view_name}/num_neighbors_1', render_ret['num_nn_1'], global_step)
        
        if self.options.TRAIN.loss_weight['boundary_loss'] != 0.:
            bd_loss = self.cal_boundary_loss(pred_pos)
            total_loss = total_loss + bd_loss * self.options.TRAIN.loss_weight['boundary_loss']
            if (global_step+1) % self.log_interval == 0:
                self.summary_writer.add_scalar(f'boudary_loss', bd_loss.item(), global_step)
        
        return total_loss
    

    def update_step(self,loss, global_step):
        grad_clip_value = self.options.TRAIN.grad_clip_value
        seperate_render_transition = self.options.TRAIN.seperate_render_transition

        if (global_step+1) % self.log_interval == 0 and grad_clip_value != 0:
            trans_grad = self.cal_grad_norm(self.transition_model)
            render_grad = self.cal_grad_norm(self.renderer)
            self.summary_writer.add_histogram('trans_grad/trans_grad_before', trans_grad, global_step)
            self.summary_writer.add_histogram('render_grad/render_grad_before', render_grad, global_step)
        
        self.optimizer.zero_grad()
        if seperate_render_transition:
            self.transition_optimizer.zero_grad()
        loss.backward()
        if grad_clip_value != 0:
            torch.nn.utils.clip_grad_norm_(self.renderer.parameters(), grad_clip_value)
            torch.nn.utils.clip_grad_norm_(self.transition_model.parameters(), grad_clip_value)
        self.optimizer.step()
        if seperate_render_transition:
            self.transition_optimizer.step()
        if self.options.TRAIN.LR.use_scheduler:
            self.optim_lr_scheduler.step()
            if seperate_render_transition:
                self.optim_lr_scheduler_transition.step()
            
        if (global_step+1) % self.log_interval == 0:
            lrs = self.get_learning_rate(self.optimizer)
            for i,lr in enumerate(lrs):
                self.summary_writer.add_scalar(f'learning_rate/lr_{i}', lr, global_step)
            if seperate_render_transition:
                lrs = self.get_learning_rate(self.transition_optimizer)
                for i,lr in enumerate(lrs):
                    self.summary_writer.add_scalar(f'learning_rate_transition/lr_{i}', lr, global_step)
        
        if (global_step+1) % self.log_interval == 0 and grad_clip_value != 0:
            trans_grad = self.cal_grad_norm(self.transition_model)
            render_grad = self.cal_grad_norm(self.renderer)
            self.summary_writer.add_histogram('trans_grad/trans_grad_after', trans_grad, global_step)
            self.summary_writer.add_histogram('render_grad/render_grad_after', render_grad, global_step)


    def eval(self, step_idx):
        """
        visulize the point cloud resutls, and the image
        """
        print('\nStep {} Eval:'.format(step_idx))
        self.eval_count += 1
        self.transition_model.eval()
        self.renderer.eval()
        view_num = len(self.test_viewnames)
        N_importance = self.options.RENDERER.ray.N_importance
        with torch.no_grad():
            dist_pred2gt_all = []
            fluid_error = FluidErrors()
            for data_idx in range(self.test_dataset_length):
                data = self.test_dataset[data_idx]
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}
                
                box = data['box']
                box_normals = data['box_normals']
                if data_idx ==0:
                    data['particles_pos'],data['particles_vel']
                    pos_for_next_step, vel_for_next_step = data['particles_pos'],data['particles_vel']
                pred_pos, pred_vel, num_fluid_nn = self.transition_model(pos_for_next_step, vel_for_next_step, box, box_normals)
                pos_for_next_step, vel_for_next_step = pred_pos.clone(), pred_vel.clone()
                
                # evaluate transition model
                pos_t1 = data['particles_pos_1']
                vel_t1 = data['particles_vel_1']
                # eval pred2gt distance
                dist_pred2gt = fluid_error.cal_errors(pred_pos.cpu().numpy(), pos_t1.cpu().numpy(), data_idx+1)
                dist_pred2gt_all.append(dist_pred2gt)
                self.summary_writer.add_scalar(f'pred2gt_distance', dist_pred2gt, self.eval_count*self.test_dataset_length+data_idx+1)
                # save to obj
                if not osp.exists(osp.join(self.particlepath, f'{step_idx}')):
                    os.makedirs(osp.join(self.particlepath, f'{step_idx}'))
                particle_name = osp.join(self.particlepath, f'{step_idx}/pred_{data_idx+1}.obj')
                with open(particle_name, 'w') as fp:
                    record2obj(pred_pos, fp, color=[255, 0, 0]) # red
                particle_name = osp.join(self.particlepath, f'{step_idx}/gt_{data_idx+1}.obj')
                with open(particle_name, 'w') as fp:
                    record2obj(pos_t1, fp, color=[3, 168, 158])
                    
                # rendering results
                # to save time, we only render several frames
                if data_idx in [0,20,30]:
                    for view_idx in range(view_num):
                        view_name = self.test_viewnames[view_idx]
                        cw = data['cw_1'][view_idx]
                        ro = self.renderer.set_ro(cw)
                        focal_length = data['focal'][view_idx]
                        rgbs = data['rgb_1'][view_idx]
                        rays = data['rays_1'][view_idx].view(-1, 6)
                        render_ret = self.render_image(pred_pos, rays.shape[0], ro, rays, focal_length, cw, iseval=True)
                        pred_rgbs_0 = render_ret['pred_rgbs_0']
                        mask_0 = render_ret['mask_0']
                        psnr_0 = mse2psnr(img2mse(pred_rgbs_0, rgbs))
                        self.summary_writer.add_scalar(f'{view_name}/psnr_{data_idx}_0', psnr_0.item(), step_idx)
                        self.visualization(pred_rgbs_0, rgbs, step_idx, mask=mask_0, prefix=f'coarse_{data_idx}_{view_name}')
                        if N_importance>0:
                            pred_rgbs_1 = render_ret['pred_rgbs_1']
                            mask_1 = render_ret['mask_1']
                            psnr_1 = mse2psnr(img2mse(pred_rgbs_1, rgbs))
                            self.summary_writer.add_scalar(f'{view_name}/psnr_{data_idx}_1', psnr_1.item(), step_idx)
                            self.visualization(pred_rgbs_1, rgbs, step_idx, mask=mask_1, prefix=f'fine_{data_idx}_{view_name}')
            self.summary_writer.add_scalar('avg_pred2gt_distance', np.mean(dist_pred2gt_all), step_idx)
        self.transition_model.train()
        self.renderer.train()

   
     