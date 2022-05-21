"""
Base Trainer
"""

import os
import random
import imageio
import numpy as np
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.loss import chamfer_distance

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
class BaseTrainer():
    def __init__(self, options):
        self.options = options
        # set random seed
        self.seed_everything(options.TRAIN.seed)
        # tools
        self.exppath = osp.join(options.expdir, options.expname)
        self.imgpath = osp.join(self.exppath, 'images')
        self.particlepath = osp.join(self.exppath, 'particles')
        os.makedirs(osp.join(self.exppath, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exppath, 'images'), exist_ok=True)
        os.makedirs(osp.join(self.exppath, 'particles'), exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=self.exppath)
        self.device = torch.device('cuda')
        self.init_fn()
        self.init_box_boundary()
        print(self.options.resume_from)
        if self.options.resume_from != '':
            self.resume(self.options.resume_from)
            
    def cal_grad_norm(self, model):
        with torch.no_grad():
            ns = []
            for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                ns.append(p.grad.detach().data.norm(2).item())
        return np.array(ns)
    
    def get_learning_rate(self,optimizer):
        lr=[]
        for param_group in optimizer.param_groups:
            lr +=[ param_group['lr'] ]
        return lr
            
    def init_fn(self):
        raise NotImplementedError()
    
    
    def resume(self):
        raise NotImplementedError()
    
    
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
        

    def load_pretained_transition_model(self, pretrained_ckpt_path, ):
        """load pretained tranistion model. 
            Note: you can rewrite it in child class if it nor meets you need.

        Args:
            pretrained_ckpt_path: path of pretrained ckpt
        """
        ckpt = torch.load(pretrained_ckpt_path)
        if 'transition_model_state_dict' in ckpt:
            ckpt = ckpt['transition_model_state_dict']
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        ckpt = {k:v for k,v in ckpt.items() if 'gravity' not in k}
        transition_model_state_dict = self.transition_model.state_dict()
        transition_model_state_dict.update(ckpt)
        self.transition_model.load_state_dict(transition_model_state_dict,strict=True)
        print(f'---> load pretrained transition model: {pretrained_ckpt_path}')
    

    def load_pretained_renderer_model(self, pretrained_ckpt_path, partial_load=False):
        """load pretrained renderer 

        Args:
            pretrained_ckpt_path: path of pretrained renderer model
            partial_load: whether only load pretrained ckpt for xyz encoding and sigma decoding parts. Defaults to False.
        """
        ckpt = torch.load(pretrained_ckpt_path)['renderer_state_dict']
        if partial_load:
            # only load xyz encoding and sigma layers
            ckpt = {k:v for k,v in ckpt.items() if 'sigma' in k or 'xyz_encoding' in k}
        render_state_dict = self.renderer.state_dict()
        
        print('---> load complete model')
        render_state_dict.update(ckpt)
        self.renderer.load_state_dict(render_state_dict, strict=True)
        print(f'---> load pretrained renderer model: {pretrained_ckpt_path}')
   
   
    def set_RGB_criterion(self):
        self.rgb_criterion = torch.nn.MSELoss().to(self.device)
        
    
    def cal_chamfer_distance(self, pred_pos, gt_pos):
        if pred_pos.ndim < 3:
            dist, _ = chamfer_distance(pred_pos.unsqueeze(0), gt_pos.unsqueeze(0))
        else:
            dist, _ = chamfer_distance(pred_pos, gt_pos)
        return dist

    
    def set_L1_criterion(self):
        self.L1_criterion = torch.nn.L1Loss()
    

    def cal_boundary_loss(self, pos):
        clipped_pos = self.strict_clip_particles(pos)
        return self.L1_criterion(pos, clipped_pos)
        
    
    def weighted_mse_loss(self, pred_pos, gt_pos, num_fluid_neighbors):
        """ calculate particle distance

        Args:
            pred_pos: predicted particle positions
            gt_pos: GT particle positions
            num_fluid_neighbors: num of neighbos, to indicate the imporance

        Returns:
            mse loss weighted by `num_fluid_neighbors`
        """
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
        dist = torch.sqrt(torch.sum((pred_pos - gt_pos)**2, dim=-1) + 1e-12)
        return torch.mean(importance * dist ** gamma)
    
    
    def ema_update_meanteacher(self, teacher, student, global_step, factor=0.99):
        # Use the true average until the exponential average is more correct
        factor = min(1 - 1 / (global_step + 1), factor)
        for ema_param, param in zip(teacher.parameters(), student.parameters()):
            ema_param.data.mul_(factor).add_(param.data, alpha=1 - factor)
    
    
    def random_sample_coords(self, H, W, global_step):
        """ sample partial pixels to be training data

        Args:
            H: image height
            W: image weight
            global_step: global training step

        Returns:
            coordinates of sampled pixels
        """
        if global_step > self.options.TRAIN.precrop_iters:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
        else:
            dH = int(H//2 * 0.5)
            dW = int(W//2 * 0.5)
            coords = torch.stack(
                                torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        return coords
 
 
    def coarse_render_image(self, particle_pos, N_ray, ro, rays, focal_length, cw, iseval=False):
        """ rendering image according `particle_pos` and others

        Args:
            particle_pos: particle positions
            N_ray: ray number
            ro: ray original point
            rays: [ray_o, ray_d]
            focal_length: focal length
            cw ([type]): camera to world transformation matrix
            iseval: whether is in evaluation. Defaults to False.

        Returns:
            rendered results 
        """
        pred_rgbs_0, pred_rgbs_1 = [], []
        num_nn_0, num_nn_1 = [], []
        mask_0, mask_1 = [], []
        for ray_idx in range(0, N_ray, self.options.RENDERER.ray.ray_chunk):
            # render
            results_i = self.renderer.coarse_rendering(particle_pos,
                                    ro,
                                    rays[ray_idx:ray_idx+self.options.RENDERER.ray.ray_chunk],
                                    focal_length,
                                    cw
                                    )
            pred_rgbs_0 += [results_i['rgb0']]
            num_nn_0 += [results_i['num_nn_0'].view(-1)]
        ret = {}
        ret['pred_rgbs_0'] = torch.cat(pred_rgbs_0, dim=0)
        ret['num_nn_0'] = torch.cat(num_nn_0, dim=0)
        return ret  
    

    def fine_render_image(self, particle_pos, N_ray, ro, rays, focal_length, cw, iseval=False):
        """ rendering image according `particle_pos` and others

        Args:
            particle_pos: particle positions
            N_ray: ray number
            ro: ray original point
            rays: [ray_o, ray_d]
            focal_length: focal length
            cw ([type]): camera to world transformation matrix
            iseval: whether is in evaluation. Defaults to False.

        Returns:
            rendered results 
        """
        pred_rgbs_0, pred_rgbs_1 = [], []
        num_nn_0, num_nn_1 = [], []
        mask_0, mask_1 = [], []
        for ray_idx in range(0, N_ray, self.options.RENDERER.ray.ray_chunk):
            # render
            results_i = self.renderer.fine_rendering(particle_pos,
                                    ro,
                                    rays[ray_idx:ray_idx+self.options.RENDERER.ray.ray_chunk],
                                    focal_length,
                                    cw
                                    )
            pred_rgbs_1 += [results_i['rgb1']]
            num_nn_1 += [results_i['num_nn_1'].view(-1)]
        ret = {}
        ret['pred_rgbs_1'] = torch.cat(pred_rgbs_1, dim=0)
        ret['num_nn_1'] = torch.cat(num_nn_1, dim=0)
        return ret     
    
    
    def render_image(self, particle_pos, N_ray, ro, rays, focal_length, cw, iseval=False):
        """ rendering image according `particle_pos` and others

        Args:
            particle_pos: particle positions
            N_ray: ray number
            ro: ray original point
            rays: [ray_o, ray_d]
            focal_length: focal length
            cw ([type]): camera to world transformation matrix
            iseval: whether is in evaluation. Defaults to False.

        Returns:
            rendered results 
        """
        pred_rgbs_0, pred_rgbs_1 = [], []
        num_nn_0, num_nn_1 = [], []
        mask_0, mask_1 = [], []
        for ray_idx in range(0, N_ray, self.options.RENDERER.ray.ray_chunk):
            # render
            results_i = self.renderer(particle_pos,
                                    ro,
                                    rays[ray_idx:ray_idx+self.options.RENDERER.ray.ray_chunk],
                                    focal_length,
                                    cw
                                    )
            pred_rgbs_0 += [results_i['rgb0']]
            num_nn_0 += [results_i['num_nn_0'].view(-1)]
            if iseval:
                mask_0 += [results_i['mask_0']]
            if self.options.RENDERER.ray.N_importance>0:
                pred_rgbs_1 += [results_i['rgb1']]
                num_nn_1 += [results_i['num_nn_1'].view(-1)]
                if iseval:
                    mask_1 += [results_i['mask_1']]
        ret = {}
        ret['pred_rgbs_0'] = torch.cat(pred_rgbs_0, dim=0)
        ret['num_nn_0'] = torch.cat(num_nn_0, dim=0)
        if iseval:
            ret['mask_0'] = torch.cat(mask_0, dim=0)
        if self.options.RENDERER.ray.N_importance>0:
            ret['pred_rgbs_1'] = torch.cat(pred_rgbs_1, dim=0)
            ret['num_nn_1'] = torch.cat(num_nn_1, dim=0)
            if iseval:
                ret['mask_1'] = torch.cat(mask_1, dim=0)
        return ret  


    def visualization(self, pred_rgbs, gt_rgbs, step, mask=None, prefix=None, pyhsical_weight_map=None):
        pred_image = self.vis_rgbs(pred_rgbs)
        gt_image = self.vis_rgbs(gt_rgbs)
        if mask is not None:
            mask = self.vis_rgbs(mask, channel=1)
            self.summary_writer.add_images(prefix.split('_')[1]+'/'+prefix+'_mask', mask.unsqueeze(0), step)
        self.summary_writer.add_images(prefix.split('_')[1]+'/'+prefix, torch.stack([gt_image, pred_image], 0), step)
        # if prefix in ['coarse', 'fine']:
        #     self.summary_writer.add_images('images', torch.stack([gt_image, pred_image], 0), step)
        # save res
        gt_rgb8 = to8b(gt_image.permute(1,2,0).detach().numpy())
        filename = '{}/{}_{:05d}.png'.format(self.imgpath, prefix, step)
        imageio.imwrite(filename, gt_rgb8)
        
        pred_rgb8 = to8b(pred_image.permute(1,2,0).detach().numpy())
        filename = '{}/{}_{:05d}_pred.png'.format(self.imgpath, prefix, step)
        imageio.imwrite(filename, pred_rgb8)
        
        if mask is not None:
            mask_rgb8 = to8b(mask.permute(1,2,0).detach().numpy())
            filename = '{}/{}_{:05d}_mask.png'.format(self.imgpath, prefix, step)
            imageio.imwrite(filename, mask_rgb8)


    def vis_rgbs(self, rgbs, channel=3):
        imgW = int(self.options.TRAIN.imgW // self.options.TRAIN.scale)
        imgH = int(self.options.TRAIN.imgH // self.options.TRAIN.scale)
        image = rgbs.reshape(imgH, imgW, channel).cpu()
        image = image.permute(2,0,1)
        return image
    

