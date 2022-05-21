"""
Evaluate render
"""

import os
import glob
import imageio
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from trainer.basetrainer import BaseTrainer

from utils.ray_utils import get_ray_directions, get_rays
from utils.particles_utils import read_obj
from models.renderer import RenderNet

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class ParticleDatset(Dataset):
    def __init__(self, particle_dir, start_index, end_index):
        self.particle_files = sorted(glob.glob(os.path.join(particle_dir, '*.npz')))[start_index:end_index]
    
    def __getitem__(self, index):
        particle_pos, _ = self._read_particles(self.particle_files[index])
        name = self.particle_files[index].split('/')[-1][:-4]
        return torch.from_numpy(particle_pos).float(), name
    
    def __len__(self,):
        return len(self.particle_files)

    def _read_particles(self, particle_path):
        """
        read initial particle information and the bounding box information
        """
        particle_info = np.load(particle_path)
        particle_pos = particle_info['pos']
        particle_vel = particle_info['vel']
        # import ipdb;ipdb.set_trace()
        particle_pos = particle_pos
        particle_vel = particle_vel
        return particle_pos, particle_vel
    

class RendererEvaluation(BaseTrainer):
    def __init__(self, options):
        self.options = options
        self.exppath = os.path.join(options.expdir, options.expname)
        os.makedirs(self.exppath, exist_ok=True)
        self.device = torch.device('cuda')
        
        self.renderer = RenderNet(self.options.RENDERER, near=self.options.TEST.near, far=self.options.TEST.far).to(self.device)
        ckpt = torch.load(self.options.resume_from)['renderer_state_dict']
        render_state_dict = self.renderer.state_dict()
        render_state_dict.update(ckpt)
        self.renderer.load_state_dict(render_state_dict, strict=True)
        print(f'---> load pretrained renderer model: {self.options.resume_from}')
        
        self.dataset = ParticleDatset(particle_dir=self.options.TEST.data_path, start_index=self.options.TEST.start_index, end_index=self.options.TEST.end_index)
        self.dataset_length = len(self.dataset)
        
    def pre_request(self):
        W, H = self.options.TEST.imgW, self.options.TEST.imgH
        focal = .5 * W / np.tan(0.5 * self.options.TEST.camera_angle_x)
        directions = get_ray_directions(H, W, focal)
        trans_matrix = np.array([
                [
                    0.3597943186759949,
                    0.09052024036645889,
                    -0.18696719408035278,
                    -4.842308521270752
                ],
                [
                    -0.2077273577451706,
                    0.15678563714027405,
                    -0.32383665442466736,
                    -8.387124061584473
                ],
                [
                    0.0,
                    0.37393447756767273,
                    0.181040421128273,
                    4.688809871673584
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])[:3, :4]
        rays_o, rays_d = get_rays(directions, torch.FloatTensor(trans_matrix))
        rays = torch.cat([rays_o, rays_d], -1)
        ret = {
                'cw': torch.from_numpy(trans_matrix).float(),
                'focal': focal,
                'rays': rays.view(-1, 6),
                }
        return ret
    

    def visulization_single_image(self, rgbs, prefix, path=None):
        image = self.vis_rgbs(rgbs)
        rgb8 = to8b(image.permute(1,2,0).detach().numpy())
        if not path:
            filename = '{}/{}.png'.format(os.path.join(self.exppath, 'render_GT'), prefix)
        else:
            filename = '{}/{}.png'.format(path, prefix)
        imageio.imwrite(filename, rgb8)


    def vis_rgbs(self, rgbs, channel=3):
        imgW = self.options.TEST.imgW
        imgH = self.options.TEST.imgH
        image = rgbs.reshape(imgH, imgW, channel).cpu()
        image = image.permute(2,0,1)
        return image
        
    
    def eval(self,):
        self.renderer.eval()
        render_params = self.pre_request()
        render_params = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in render_params.items()}
        cw  = render_params['cw'].to(self.device)
        focal_length = render_params['focal']
        rays = render_params['rays'].to(self.device)
        
        render_GT_dir = os.path.join(self.exppath, 'render_GT')
        if not os.path.exists(render_GT_dir):
            os.makedirs(render_GT_dir)
        render_predpos_dir = os.path.join(self.exppath, 'render_PredPos')
        if not os.path.exists(render_predpos_dir):
            os.makedirs(render_predpos_dir)
        with torch.no_grad():
            for data_idx in tqdm(range(self.dataset_length), total=self.dataset_length):
                if data_idx > 52:
                    break
                gt_pos, name = self.dataset[data_idx]
                gt_pos = gt_pos.to(self.device)
                
                ro = self.renderer.set_ro(cw)
                render_ret = self.render_image(gt_pos, rays.shape[0], ro, rays, focal_length, cw, iseval=True)
                pred_rgbs_0 = render_ret['pred_rgbs_0']
                self.visulization_single_image(pred_rgbs_0, prefix=f'coarse_pred_{name}')
                if self.options.RENDERER.ray.N_importance>0:
                    pred_rgbs_1 = render_ret['pred_rgbs_1']
                    self.visulization_single_image(pred_rgbs_1, prefix=f'fine_pred_{name}')
            
            # pred_files = sorted(glob.glob(os.path.join('119999', 'pred_*.obj')))
            # for file in tqdm(pred_files):
            #     pred_pos = read_obj(file)
            #     pred_pos = torch.Tensor(pred_pos).to(self.device)

            #     name = file.split('/')[-1][5:-4]
            #     ro = self.renderer.set_ro(cw)
            #     render_ret = self.render_image(pred_pos, rays.shape[0], ro, rays, focal_length, cw, iseval=True)
            #     pred_rgbs_0 = render_ret['pred_rgbs_0']
            #     self.visulization_single_image(pred_rgbs_0, prefix=f'coarse_pred_{name}', path=render_predpos_dir)
            #     if self.options.RENDERER.ray.N_importance>0:
            #         pred_rgbs_1 = render_ret['pred_rgbs_1']
            #         self.visulization_single_image(pred_rgbs_1, prefix=f'fine_pred_{name}', path=render_predpos_dir)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    from configs import warmup_training_config
    cfg = warmup_training_config()
    evaluator = RendererEvaluation(cfg)
    evaluator.eval()
    