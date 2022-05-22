"""
Training an renderer

Given P0, finetune renderer
"""


import numpy as np
import os.path as osp
from tqdm import tqdm

import torch

from .basetrainer import BaseTrainer
from models.renderer import RenderNet
from datasets.dataset import BlenderDataset
from utils.lr_schedulers import ExponentialLR

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())

class Trainer(BaseTrainer):
    def init_fn(self):
        self.start_step = 0
        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.set_RGB_criterion()


    def build_dataloader(self):
        self.train_view_names = self.options['train'].views.warmup
        self.test_viewnames = self.options['test'].views
        self.dataset = BlenderDataset(self.options.train.path, self.options,
                                            imgW=self.options.TRAIN.imgW, imgH=self.options.TRAIN.imgH,
                                            imgscale=self.options.TRAIN.scale, viewnames=self.train_view_names, split='train')
        self.dataset_length = len(self.dataset)
        self.test_dataset = BlenderDataset(self.options.test.path, self.options, 
                                            imgW=self.options.TEST.imgW, imgH=self.options.TEST.imgH,
                                            imgscale=self.options.TEST.scale, viewnames=self.test_viewnames, split='test')
        self.test_dataset_length = len(self.test_dataset)
        print('---> dataloader has been build')
    
    
    def build_model(self):
        # build model
        self.renderer = RenderNet(self.options.RENDERER, near=self.options.near, far=self.options.far).to(self.device)
        # load pretrained checkpoints
        if self.options.TRAIN.pretained_renderer != '':
            self.load_pretained_renderer_model(self.options.TRAIN.pretained_renderer, partial_load=self.options.TRAIN.partial_load)


    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.renderer.parameters(), lr=self.options.TRAIN.LR.lr)
        if self.options.TRAIN.LR.use_scheduler:
            self.lr_scheduler = ExponentialLR(self.optimizer, decay_epochs=self.options.TRAIN.LR.decay_epochs ,gamma=0.1)
            
    
    def resume(self, ckpt_file):
        checkpoint = torch.load(ckpt_file)
        self.start_step = checkpoint['step']
        self.renderer.load_state_dict(checkpoint['renderer_state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=True)
        
        
    def save_checkpoint(self, global_step):
        torch.save({'step':global_step,
                    'renderer_state_dict':self.renderer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, 
                    osp.join(self.exppath, 'models', f'{global_step}.pt'))  
        
    
    def train(self):
        view_num = len(self.train_view_names)
        H = int(self.options.TRAIN.imgH // self.options.TRAIN.scale)
        W = int(self.options.TRAIN.imgW // self.options.TRAIN.scale)
        self.renderer.train()
        for step_idx in tqdm(range(self.start_step, self.options.TRAIN.N_iters), total=self.options.TRAIN.N_iters, desc='Iteration:'):
            data_idx = 0
            data = self.dataset[data_idx]
            data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}

            loss = self.train_step(data, view_num, H, W, step_idx)
            self.update_step(loss)
                     
            # evaluation
            if (step_idx+1) % self.options.TRAIN.save_interval == 0:
                self.eval(step_idx)
                self.save_checkpoint(step_idx)
                    
                    
    def update_step(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.options.TRAIN.LR.use_scheduler:
            self.lr_scheduler.step()
    
        
    def train_step(self, data, view_num, H, W, step_idx):
        # -------
        # render by a nerf model, and then calculate mse loss
        # -------
        ray_chunk = self.options.RENDERER.ray.ray_chunk
        N_importance = self.options.RENDERER.ray.N_importance
        total_loss = 0.
        for view_idx in range(view_num):
            gt_pos = data['particles_pos']
            view_name = self.train_view_names[view_idx]
            cw_t1 = data['cw'][view_idx]
            rgbs_t1 = data['rgb'][view_idx]
            rays_t1 = data['rays'][view_idx]
            focal_length = data['focal'][view_idx]
            # randomly sample pixel
            coords = self.random_sample_coords(H,W,step_idx)
            coords = torch.reshape(coords, [-1,2])
            select_inds = np.random.choice(coords.shape[0], size=[ray_chunk], replace=False)
            select_coords = coords[select_inds].long()
            rays_t1 = rays_t1[select_coords[:, 0], select_coords[:, 1]]
            rgbs_t1 = rgbs_t1.view(H, W, -1)[select_coords[:, 0], select_coords[:, 1]]
            # render
            ro_t1 = self.renderer.set_ro(cw_t1)
            render_ret = self.render_image(gt_pos, ray_chunk, ro_t1, rays_t1, focal_length, cw_t1)
            # calculate mse loss
            rgbloss_0 = self.rgb_criterion(render_ret['pred_rgbs_0'], rgbs_t1[:ray_chunk])
            if N_importance>0:
                rgbloss_1 = self.rgb_criterion(render_ret['pred_rgbs_1'], rgbs_t1[:ray_chunk])
                rgbloss = rgbloss_0 + rgbloss_1
            else:
                rgbloss = rgbloss_0
            total_loss = total_loss+rgbloss
                    
            # log
            if (step_idx+1) % self.options.log_interval == 0:
                self.summary_writer.add_scalar(f'{view_name}/rgbloss_0', rgbloss_0.item(), step_idx)
                self.summary_writer.add_histogram(f'{view_name}/num_neighbors_0', render_ret['num_nn_0'], step_idx)
                if N_importance>0:
                    self.summary_writer.add_scalar(f'{view_name}/rgbloss_1', rgbloss_1.item(), step_idx)
                    self.summary_writer.add_histogram(f'{view_name}/num_neighbors_1', render_ret['num_nn_1'], step_idx)
                self.summary_writer.add_scalar(f'{view_name}/rgbloss', rgbloss.item(), step_idx)
        return total_loss
    
        
    def eval(self, step_idx):
        print('\n Eval:', step_idx)
        N_importance = self.options.RENDERER.ray.N_importance
        self.renderer.eval()
        view_num = len(self.test_viewnames)
        with torch.no_grad():
            for  data_idx in [0]:
                data = self.test_dataset[data_idx]
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}
                gt_pos = data['particles_pos']
                for view_idx in range(view_num):
                    view_name = self.test_viewnames[view_idx]
                    cw = data['cw'][view_idx]
                    ro = self.renderer.set_ro(cw)
                    focal_length = data['focal'][view_idx]
                    rgbs = data['rgb'][view_idx]
                    rays = data['rays'][view_idx].view(-1, 6)
                    render_ret = self.render_image(gt_pos, rays.shape[0], ro, rays, focal_length, cw, iseval=True)
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
        self.renderer.train()
