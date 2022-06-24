"""
The whole framework
"""

import os
import joblib
import numpy as np
import os.path as osp
from tqdm import tqdm
import imageio

import torch

from trainer.basetrainer import BaseTrainer
from models.renderer import RenderNet
from models.transmodel import ParticleNet
from datasets.dataset import BlenderDataset
from utils.particles_utils import record2obj
from utils.point_eval import FluidErrors


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Evaluator(BaseTrainer):
    def init_fn(self):
        self.start_step = 0
        self.eval_count = 0
        self.build_dataloader()
        self.build_model()

    def build_dataloader(self):
        self.test_viewnames = self.options['test'].views
        self.test_dataset = BlenderDataset(self.options.test.path, self.options, 
                                            start_index=self.options['test'].start_index, end_index=self.options['test'].end_index,
                                            imgW=self.options.TEST.imgW, imgH=self.options.TEST.imgH,
                                            imgscale=self.options.TEST.scale,viewnames=self.test_viewnames, split='test')
        self.test_dataset_length = len(self.test_dataset)
        print('---> dataloader has been build')
       
        
    def build_model(self):
        # build model
        gravity = self.options.gravity
        print('---> set gravity', gravity)
        self.transition_model = ParticleNet(gravity=gravity).to(self.device)
        self.renderer = RenderNet(self.options.RENDERER, near=self.options.near, far=self.options.far).to(self.device)
    
    def resume(self, ckpt_file):
        # resume
        checkpoint = torch.load(ckpt_file)
        self.renderer.load_state_dict(checkpoint['renderer_state_dict'], strict=True)
        self.transition_model.load_state_dict(checkpoint['transition_model_state_dict'], strict=True)
        print('---> model has been resumed from {}'.format(ckpt_file))

              
    def eval(self):
        """
        visulize the point cloud resutls, and the image
        """
        self.transition_model.eval()
        self.renderer.eval()
        view_num = len(self.test_viewnames)
        with torch.no_grad():
            dist_pred2gt_all = []
            self.fluid_error = FluidErrors()
            for data_idx in tqdm(range(self.test_dataset_length), total=self.test_dataset_length):
                data = self.test_dataset[data_idx]
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}
                
                box = data['box']
                box_normals = data['box_normals']
                if data_idx ==0:
                    data['particles_pos'],data['particles_vel']
                    pos_for_next_step, vel_for_next_step = data['particles_pos'],data['particles_vel']
                pred_pos, pred_vel, num_fluid_nn = self.transition_model(pos_for_next_step, vel_for_next_step, box, box_normals)
                pos_for_next_step, vel_for_next_step = pred_pos.clone(), pred_vel.clone()
                
                # --------
                # evaluate transition model
                # --------
                pos_t1 = data['particles_pos_1']
                vel_t1 = data['particles_vel_1']
                # eval pred2gt distance
                dist_pred2gt = self.fluid_error.cal_errors(pred_pos.cpu().numpy(), pos_t1.cpu().numpy(), data_idx+1)
                dist_pred2gt_all.append(dist_pred2gt)
                # save to obj
                if not osp.exists(osp.join(self.particlepath, 'Pred')):
                    os.makedirs(osp.join(self.particlepath, 'Pred'))
                    os.makedirs(osp.join(self.particlepath, 'GT'))
                particle_name = osp.join(self.particlepath, f'Pred/{data_idx+1}.obj')
                with open(particle_name, 'w') as fp:
                    record2obj(pred_pos, fp, color=[255, 0, 0]) # red
                particle_name = osp.join(self.particlepath, f'GT/{data_idx+1}.obj')
                with open(particle_name, 'w') as fp:
                    record2obj(pos_t1, fp, color=[3, 168, 158])
                    
                # --------
                # evaluate renderer 
                # --------
                for view_idx in range(view_num):
                    view_name = self.test_viewnames[view_idx]
                    cw = data['cw_1'][view_idx]
                    ro = self.renderer.set_ro(cw)
                    focal_length = data['focal'][view_idx]
                    rgbs = data['rgb_1'][view_idx]
                    rays = data['rays_1'][view_idx].view(-1, 6)

                    render_ret = self.render_image(pred_pos, rays.shape[0], ro, rays, focal_length, cw, iseval=True)
                    assert self.renderer.cfg.use_mask == True
                    # if data_idx >= 50:
                    #     self.renderer.cfg.use_mask = True
                    pred_rgbs_0 = render_ret['pred_rgbs_0']
                    prefix=f'coarse/{view_name}'
                    self.visualization(pred_rgbs_0, rgbs, prefix, data_idx+1)
                    if self.options.RENDERER.ray.N_importance>0:
                        pred_rgbs_1 = render_ret['pred_rgbs_1']
                        prefix=f'fine/{view_name}'
                        self.visualization(pred_rgbs_1, rgbs, prefix, data_idx+1)
            
            print('----------------- trained 50 steps ------------------------')
            print('Pred2GT:', np.mean(dist_pred2gt_all[0:49]))
            print('Pred2GT-10:', np.mean(dist_pred2gt_all[:10]))
            print('Pred2GT-end:', dist_pred2gt_all[48])
            
            print('\n----------------- rollout 10 steps ------------------------')
            print('Pred2GT:', np.mean(dist_pred2gt_all[-10:]))
            print('Pred2GT-5:', np.mean(dist_pred2gt_all[-5]))
            print('Pred2GT-end:', dist_pred2gt_all[-1])

            joblib.dump({'dist': dist_pred2gt_all}, osp.join(self.exppath, 'pred2gt.pt'))
        self.transition_model.train()
        self.renderer.train()
        

    def visualization(self, pred_rgbs, gt_rgbs,prefix=None, data_idx=0):
        pred_image = self.vis_rgbs(pred_rgbs)
        gt_image = self.vis_rgbs(gt_rgbs)
        
        if not os.path.exists(osp.join(self.imgpath, prefix)):
            os.makedirs(osp.join(self.imgpath, prefix, 'GT'))
            os.makedirs(osp.join(self.imgpath, prefix, 'Pred'))
        
        # save res
        gt_rgb8 = to8b(gt_image.permute(1,2,0).detach().numpy())
        filename = '{}/{}/GT/{:05d}.png'.format(self.imgpath, prefix, data_idx)
        imageio.imwrite(filename, gt_rgb8)
        
        pred_rgb8 = to8b(pred_image.permute(1,2,0).detach().numpy())
        filename = '{}/{}/Pred/{:05d}.png'.format(self.imgpath, prefix, data_idx)
        imageio.imwrite(filename, pred_rgb8)
        

    def vis_rgbs(self, rgbs, channel=3):
        imgW = int(self.options.TEST.imgW // self.options.TEST.scale)
        imgH = int(self.options.TEST.imgH // self.options.TEST.scale)
        image = rgbs.reshape(imgH, imgW, channel).cpu()
        image = image.permute(2,0,1)
        return image

   
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    from configs import end2end_training_config, dataset_config

    cfg_datasets = dataset_config()
    cfg_e2e = end2end_training_config()

    cfg_dataset = cfg_datasets[cfg_e2e.dataset]
    cfg_e2e.update(cfg_dataset)

    evaluator = Evaluator(cfg_e2e)
    evaluator.eval()
    