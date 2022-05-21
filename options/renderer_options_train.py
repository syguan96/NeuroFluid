import os
import json
import argparse
import os.path as osp
from collections import namedtuple

from .base_options import BaseTrainOptions

class TrainOptions(BaseTrainOptions):
    def init_options(self):

        data = self.parser.add_argument_group('configurations related to data')
        data.add_argument('--data_path', type=str, default='data/my-1230-honey-rot-processed', help='data dir')
        data.add_argument('--data_type', type=str, choices=['blender', 'splishsplash'])
        data.add_argument('--imgW', type=int, default=400, help='image width')
        data.add_argument('--imgH', type=int, default=400, help='image height')
        data.add_argument('--near', type=float, default=9.0, help='near bound')
        data.add_argument('--far', type=float, default=13.0, help='far bound')
        
        renderer = self.parser.add_argument_group('Training Renderer configuration')
        renderer.add_argument('--N_importance', type=int, default=0)
        renderer.add_argument('--scale', type=int, default=1)
        renderer.add_argument('--ray_chunk', type=int, default=1024, help='chunks for ray')
        renderer.add_argument('--num_neighbor', type=int, default=500, help='MAX num of neighbors')
        renderer.add_argument('--num_particles_every_ray', type=int, default=64, help='num of coarsely sampled points on each ray')
        renderer.add_argument('--precrop_iters', type=float, default=500)
        renderer.add_argument('--num_sampled_ray', type=int, default=1024, help='number of sampled ray to train')
        renderer.add_argument('--fix_radius', type=int, default=0, help='seach nn within a fixed raduis sphere')
        renderer.add_argument('--particle_radius', type=int, default=0.025, help='particle radius')
        renderer.add_argument('--search_raduis_scale', type=float, default=1.25, help='search raduis, search in raduis (L2)')
        renderer.add_argument('--encoding_density', type=int, default=0)
        renderer.add_argument('--encoding_smoothed_pos', type=int, default=0)
        renderer.add_argument('--encoding_smoothed_dir', type=int, default=0)
        renderer.add_argument('--encoding_pca_dirs', type=int, default=0)
        renderer.add_argument('--encoding_var', type=int, default=0)
        renderer.add_argument('--exclude_ray_when_smoothing', type=int, default=0)
        renderer.add_argument('--same_smooth_factor', type=int, default=0)
        
        train = self.parser.add_argument_group('Training configuration')
        train.add_argument('--seed', type=int, default=10, help='random seed')
        train.add_argument('--N_iters', type=int, default=100000, help='Total iteration steps')
        train.add_argument('--render_lr', type=float, default=1e-4, help='initial learning rate')
        train.add_argument('--pretained_renderer', type=str, default=None)
        train.add_argument('--only_load_partial_render', type=int, default=0)
        train.add_argument('--seperately_training_renderer', type=int, default=0)
        train.add_argument('--viewnames', type=str, default=None, help='view1,view2')
        train.add_argument('--test_viewnames', type=str, default=None, help='view1,view2')
        train.add_argument('--mv_training', type=int, default=0, help='training multi-view images in one training step')
        train.add_argument('--grad_clip_value', type=float, default=0, help='0: no gradient clip')
        train.add_argument('--use_lr_scheduler', type=int, default=0)
        train.add_argument('--start_index', type=int, default=0)
        train.add_argument('--end_index', type=int, default=50)
        train.add_argument('--use_mask', type=int, default=0)
        train.add_argument('--decay_epochs', type=int, default=100000, help='decay_epochs')
        
        hyparam_hy = self.parser.add_argument_group('hyper params created by hy')
        # train.add_argument('--pos_num', type=int, default=3000, help='opt pos num')
        hyparam_hy.add_argument('--pos_num', type=int, default=3000, help='opt pos num')
        hyparam_hy.add_argument('--pos_lr', type=float, default=1e-5, help='pos lr')
        hyparam_hy.add_argument('--optimize_frame_pos', type=int, default=-1, help='optpos for single frame or sequence')
        hyparam_hy.add_argument('--opt_iters', type=int, default=500, help='opt iters gap between renderer and pos')
        
        hyparam_hy.add_argument('--pos_iterations', type=int, default=10, help='opt iters gap between renderer and pos')
        hyparam_hy.add_argument('--transition_iterations', type=int, default=10, help='opt iters gap between renderer and pos')