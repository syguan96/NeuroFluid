"""
Save particles to obj/ply
"""
import os
import glob
import pickle as pkl
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--dst_path', type=str, default='')
parser.add_argument('--record_type', default='ply', choices=['ply', 'obj'])


def read_obj(file):
    with open(file, 'r') as f:
        pos = []
        while True:
            line = f.readline()
            if not line:
                break
            
            strs = line.split(' ')
            if strs[0] == 'v':
                pos.append([float(strs[1]), float(strs[2]), float(strs[3])])
    
    return pos

def record2ply(particles, fp):
    assert particles.shape[-1] == 3
    for i in range(particles.shape[0]):
        info = '{:.2f} {:.2f} {:.2f}\n'.format(particles[i][0], particles[i][1], particles[i][2])
        fp.write(info)
    return None

def record2obj(particles, fp, color=[255, 0, 0]):
    for i in range(particles.shape[0]):
        info = 'v {:.2f} {:.2f} {:.2f} {} {} {}\n'.format(particles[i][0], particles[i][1], particles[i][2], color[0], color[1], color[2])
        fp.write(info)
    return None

def main(options):
    particles_name = glob.glob(osp.join(options.data_path, '*.pkl'))
    for i in tqdm(range(len(particles_name))):
        with open(particles_name[i], 'rb') as f:
            particle_data = pkl.load(f)
            location = np.array(particle_data['location']).reshape(-1, 3)
            if options.record_type == 'ply':
                fp = open(osp.join(options.dst_path, osp.basename(particles_name[i])[:-4]+'.ply'), 'w')
                fp.write('ply\n')
                fp.write('format ascii 1.0\n')
                fp.write('element vertex {}\n'.format(location.shape[0]))
                fp.write('property float32 x\n')
                fp.write('property float32 y\n')
                fp.write('property float32 z\n')
                fp.write('end_header\n')
                record2ply(location, fp)
                fp.close()
            else:
                fp = open(osp.join(options.dst_path, osp.basename(particles_name[i])[:-4]+'.obj'), 'w')
                fp.close()

if __name__ == '__main__':
    options = parser.parse_args()
    if not osp.exists(options.dst_path):
        os.makedirs(options.dst_path)
    main(options)