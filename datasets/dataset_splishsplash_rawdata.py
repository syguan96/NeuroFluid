"""
Load data from splishsplash
"""
import sys
sys.path.append('..')

import os
import json
import glob
import pickle as pkl
import joblib
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ParticleDataset(Dataset):
    def __init__(self, data_path, data_type, start, end, random_rot=True, window=3):
        """
        window: 3
            f0 --> f1 --> f2
        """
        super(ParticleDataset, self).__init__()
        self.random_rot = random_rot
        self.window = window
        self.root_dir = data_path
        self.start = start
        self.end = end
        if data_type == 'raw':
            self.dataitems = self.collect_particles_raw()
        elif data_type == 'blender':
            self.dataitems = self.collect_particles_blender()
        elif data_type == 'blender_all':
            self.dataitems = self.collect_particles_blender_all()
        print('Total lens:',len(self.dataitems) )


    def _read_box(self, box_path):
        # bbox_path = self.meta['bounding_box']
        box_info = joblib.load(box_path)
        box = box_info['box']
        box_normals = box_info['box_normals']
        box = box
        box_normals = box_normals
        return box, box_normals


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

    
    def collect_particles_blender(self):
        samples = []
        particle_paths_i = glob.glob(osp.join(self.root_dir, 'view_0/train/particles/*.npz'))
        particle_paths_i.sort(key=lambda x:int(x.split('/')[-1][:-4]))
        particle_paths_i = particle_paths_i[self.start:self.end]
        box_path = osp.join(self.root_dir, 'box.pt')
        box, box_normal = self._read_box(box_path)
        for idx in range(len(particle_paths_i)-self.window+1):
            sample = {
                            'box': box,
                            'box_normals': box_normal,
                            }
            for ii in range(self.window):
                _pos, _vel = self._read_particles(particle_paths_i[idx+ii])
                sample[f'particles_pos_{ii}'] = _pos
                sample[f'particles_vel_{ii}'] = _vel
            samples.append(sample)
        return samples

    def collect_particles_blender_all(self):
        particles_dirs = glob.glob(osp.join(self.root_dir, '*'))
        samples = []
        for particles_dir in particles_dirs:
            particle_paths_i = glob.glob(osp.join(particles_dir, 'train/particles/*.npz'))
            particle_paths_i.sort(key=lambda x:int(x.split('/')[-1][:-4]))
            particle_paths_i = particle_paths_i[self.start:self.end]
            box_path = osp.join(self.root_dir, 'box.pt')
            box, box_normal = self._read_box(box_path)
            for idx in range(len(particle_paths_i)-self.window+1):
                sample = {
                                'box': box,
                                'box_normals': box_normal,
                                }
                for ii in range(self.window):
                    _pos, _vel = self._read_particles(particle_paths_i[idx+ii])
                    sample[f'particles_pos_{ii}'] = _pos
                    sample[f'particles_vel_{ii}'] = _vel
                samples.append(sample)
        return samples

    def collect_particles_raw(self,):
        particles_dirs = glob.glob(osp.join(self.root_dir, 'sim*'))
        samples = []
        for particles_dir in particles_dirs:
            particle_paths_i = glob.glob(osp.join(particles_dir, 'output/fluid_*.npz'))
            particle_paths_i.sort(key=lambda x:int(x.split('_')[-1][:-4]))
            particle_paths_i = particle_paths_i[self.start:self.end]
            box_path = osp.join(particles_dir, 'box.pt')
            box, box_normal = self._read_box(box_path)
            for idx in range(len(particle_paths_i)-self.window):
                sample = {
                            'box': box,
                            'box_normals': box_normal,
                            }
                for ii in range(self.window):
                    _pos, _vel = self._read_particles(particle_paths_i[idx+ii])
                    sample[f'particles_pos_{ii}'] = _pos
                    sample[f'particles_vel_{ii}'] = _vel
                samples.append(sample)
        return samples


    def __getitem__(self, index):
        data = self.dataitems[index]
        returned_data = {}
        if self.random_rot:
            angle = np.random.uniform(0, 2*np.pi)
            s = np.sin(angle)
            c = np.cos(angle)
            # rot z angle
            rand_R = np.array([c, -s, 0, s, c, 0, 0, 0, 1], dtype=np.float32).reshape((3,3))
            for k,v in data.items():
                returned_data[k] = torch.from_numpy(np.matmul(v, rand_R)).float()
        else:
            for k,v in data.items():
                returned_data[k] = torch.from_numpy(v).float()
        return returned_data
        

    def __len__(self):
        return len(self.dataitems)


if __name__ == '__main__':
    dataset = ParticleDataset()
    print('Done')