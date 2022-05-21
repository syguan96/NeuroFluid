"""
Load data exported from blender for the renderer
"""
import sys
sys.path.append('..')

import os
import json
import pickle as pkl
import joblib
import numpy as np
import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils.ray_utils import get_ray_directions, get_rays

class BlenderDataset(Dataset):
    def __init__(self, root_dir, cfg, imgW, imgH, imgscale, viewnames, split='train'):
        super(BlenderDataset, self).__init__()
        self.data_type = cfg.data_type
        # self.half_res = cfg.half_res
        self.viewnames = viewnames
        self.cfg = cfg
        self.split = split
        self.img_wh = (imgW, imgH)
        self.img_scale = imgscale
        assert self.img_wh[0] == self.img_wh[1], 'image width should be equal to image height'
        self.root_dir = root_dir #cfg.data_path
        self.transforms = T.ToTensor()
        # self.view_num = len(self.viewnames)
        self.read_metas(self.viewnames)
        self.read_box()
        print('Total dataset size:', self.all_rgbs_mv.shape[1])

    def read_metas(self, viewnames):
        self.all_rays_mv, self.all_rgbs_mv, self.all_cw_mv, self.focal_mv, self.particles_poss_mv, self.particles_vels_mv = [], [], [], [], [], []
        for iii, viewname in enumerate(viewnames):
            _root_dir = osp.join(self.root_dir, viewname)
            all_rays_i, all_rgbs_i, all_cw_i, focal_i, particles_poss_i, particles_vels_i = self._read_meta(_root_dir)
            self.all_rays_mv.append(all_rays_i)
            self.all_rgbs_mv.append(all_rgbs_i)
            self.all_cw_mv.append(all_cw_i)
            self.focal_mv.append(focal_i)
            if iii == 0:
                self.particles_poss_mv.append(np.stack(particles_poss_i, 0))
                self.particles_vels_mv.append(np.stack(particles_vels_i, 0))
        self.all_rays_mv = np.stack(self.all_rays_mv, 0)
        self.all_rgbs_mv = np.stack(self.all_rgbs_mv, 0)
        self.all_cw_mv = np.stack(self.all_cw_mv, 0)
        # self.focal_mv = np.array(self.focal_mv)
        self.particles_poss_mv = np.stack(self.particles_poss_mv, 0)
        self.particles_vels_mv = np.stack(self.particles_vels_mv, 0)
        # import ipdb;ipdb.set_trace()
        

    def _read_meta(self, root_dir):
        """
        read meta file. output rays and rgbs
        """
        with open(os.path.join(root_dir, f'transforms_{self.split}.json'), 'r') as f:
            self.meta = json.load(f)
    
        # parse
        # if self.half_res:
        #     W, H = self.img_wh[0] //2, self.img_wh[1] //2
        # else:
        #     W, H = self.img_wh
        W, H = int(self.img_wh[0] // self.img_scale), int(self.img_wh[1] // self.img_scale)
        focal = .5 * W / np.tan(0.5 * self.meta['camera_angle_x'])
        # get ray direction for all pixels
        directions = get_ray_directions(H, W, focal)
        image_paths = []
        poses = []
        all_rays = []
        all_rgbs = []
        all_cw = []
        # particles_path = []
        particle_poss = []
        particle_vels = []
        # self.all_mask = []
        for frame in self.meta['frames'][self.cfg.start_index:self.cfg.end_index]:
            # get particles
            # particles_path.append(frame['particle_path'])
            if len(self.particles_poss_mv) == 0:
                particle_pos, particle_vel = self._read_particles(osp.join(root_dir, self.split, frame['particle_path']))
                particle_poss.append(particle_pos)
                particle_vels.append(particle_vel)
            # get orignal point and directrion
            pose = np.array(frame['transform_matrix'])[:3, :4]
            poses.append(pose)
            c2w = torch.FloatTensor(pose)
            all_cw.append(pose)
            rays_o, rays_d = get_rays(directions, c2w)
            all_rays += [torch.cat([rays_o, rays_d], -1).numpy()]
            # read images
            image_path = osp.join(root_dir, '{}.png'.format(frame['file_path']))
            image_paths.append(image_path)
            image = Image.open(image_path)
            # if self.half_res:
            image = image.resize((int(self.img_wh[0]// self.img_scale), int(self.img_wh[1]// self.img_scale)), Image.ANTIALIAS)
            image = (np.asarray(image))/ 255.
            image = image.reshape(-1, 4)
            image = image[:, :3]*image[:, -1:] + (1-image[:, -1:])
            # image = self.transforms(image)
            # image = image.view(4, -1).permute(1,0) #(H*W, 4), RGBA image
            # image = image[:, :3]*image[:, -1:] + (1-image[:, -1:]) # blend A to RGB, assume white background. 
            all_rgbs.append(image)
        all_rays = np.stack(all_rays, 0)
        all_rgbs = np.stack(all_rgbs, 0)
        all_cw = np.stack(all_cw, 0)
        return all_rays, all_rgbs, all_cw, focal, particle_poss, particle_vels
        # return all_rays, all_rgbs, all_cw, focal, particles_path


    def read_box(self):
        bbox_path = self.meta['bounding_box']
        box_info = joblib.load(osp.join(self.root_dir, bbox_path))
        self.box = box_info['box']
        self.box_normals = box_info['box_normals']


    def _read_particles(self, particle_path):
        """
        read initial particle information and the bounding box information
        """
        if self.data_type == 'blender':
            # particle_info = np.load(osp.join(self.root_dir, self.split, particle_path))
            # with open(osp.join(self.root_dir, self.split, particle_path), 'rb') as fp:
            with open(particle_path, 'rb') as fp:
                particle_info = pkl.load(fp)
            particle_pos = np.array(particle_info['location']).reshape(-1, 3)
            particle_vel = np.array(particle_info['velocity']).reshape(-1, 3)
        elif self.data_type == 'splishsplash':
            # particle_info = np.load(osp.join(self.root_dir, self.split, particle_path))
            particle_info = np.load(particle_path)
            particle_pos = particle_info['pos']
            particle_vel = particle_info['vel']
        else:
            raise NotImplementedError('please enter correct data type')
        # import ipdb;ipdb.set_trace()
        # particle_pos = torch.from_numpy(particle_pos).float()
        # particle_vel = torch.from_numpy(particle_vel).float()
        return particle_pos, particle_vel


    def __getitem__(self, index):
        # rays = self.all_rays_mv[:, index]
        # rgbs = self.all_rgbs_mv[:, index]
        data = {}
        data['cw'] = torch.from_numpy(self.all_cw_mv[:,index]).float()
        data['rgb'] = torch.from_numpy(self.all_rgbs_mv[:, index]).float()
        data['rays'] = torch.from_numpy(self.all_rays_mv[:, index]).float()
        data['box'] = torch.from_numpy(self.box).float()
        data['box_normals'] = torch.from_numpy(self.box_normals).float()
        data['particles_pos'] = torch.from_numpy(self.particles_poss_mv[0, index]).float()
        data['particles_vel'] = torch.from_numpy(self.particles_vels_mv[0, index]).float()
        data['focal'] = self.focal_mv
        # data['view_name'] = self.viewnames
        # if index < self.all_rgbs_mv.shape[1]:
        data['cw_1'] = torch.from_numpy(self.all_cw_mv[:,index+1]).float()
        data['rays_1'] = torch.from_numpy(self.all_rays_mv[:, index+1]).float()
        data['rgb_1'] = torch.from_numpy(self.all_rgbs_mv[:, index+1]).float()
        data['particles_pos_1'] = torch.from_numpy(self.particles_poss_mv[0, index+1]).float()
        data['particles_vel_1'] = torch.from_numpy(self.particles_vels_mv[0, index+1]).float()
        return data

    def __len__(self):
        return self.all_rgbs_mv.shape[1]-1


if __name__ == '__main__':
    dataset = BlenderDataset()
    print('Done')