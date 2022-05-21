"""
training the transition model
"""

import os
import random
import numpy as np
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.transmodel import ParticleNet
from datasets.dataset_splishsplash_rawdata import ParticleDataset
# from datasets.dataset import BlenderDataset as evalBlenderDataset
from utils.particles_utils import record2obj
from utils.point_eval import FluidErrors


class Trainer():
    def __init__(self, options):
        self.options = options
        self.seed_everything(options.TRAIN.seed)
        self.device = torch.device('cuda')

        # logger
        self.exppath = osp.join(options.expdir, options.expname)
        self.particlepath = osp.join(self.exppath, 'particles')
        self.modelpath = osp.join(self.exppath, 'particles')
        os.makedirs(osp.join(self.exppath, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exppath, 'particles'), exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=self.exppath)
        
        self.init_fn()
        self.init_box_boundary()

        if self.options.resume_from != '':
            self.resume(self.options.resume_from)

    def init_fn(self):
        self.eval_count = 0
        self.start_step = 0
        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.L1_criterion = torch.nn.L1Loss()


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


    def init_box_boundary(self):
        particle_radius = self.options.TRAIN.particle_radius
        self.x_bound = [1-particle_radius, -1+particle_radius]
        self.y_bound = [1-particle_radius, -1+particle_radius]
        self.z_bound = [2.4552-particle_radius, -1+particle_radius]


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

    
    def strict_clip_particles(self, pos):
        assert len(pos.shape) == 2
        clipped_x = torch.clamp(pos[:, 0], max=self.x_bound[0], min=self.x_bound[1])
        clipped_y = torch.clamp(pos[:, 1], max=self.y_bound[0], min=self.y_bound[1])
        clipped_z = torch.clamp(pos[:, 2], max=self.z_bound[0], min=self.z_bound[1])
        clipped_pos = torch.stack((clipped_x, clipped_y, clipped_z), dim=1)
        return clipped_pos

    def cal_boundary_loss(self, pos):
        clipped_pos = self.strict_clip_particles(pos)
        return self.L1_criterion(pos, clipped_pos)

        
    def cal_grad_norm(self, model):
        ns = []
        for p in list(filter(lambda p: p.grad is not None, model.parameters())):
            ns.append(p.grad.detach().data.norm(2).item())
        return np.array(ns)
                
                
    def resume(self, ckpt_file):
        print('resume from', ckpt_file)
        checkpoint = torch.load(ckpt_file)
        self.transition_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    
    def build_dataloader(self):
        self.dataset = ParticleDataset(self.options.TRAIN.datapath.train,
                                       data_type=self.options.TRAIN.datapath.train_datatype, 
                                       start=self.options.TRAIN.start_index,
                                       end=self.options.TRAIN.end_index,
                                       random_rot=True, window=3)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=8)
        self.dataset_length = len(self.dataset)
        self.test_dataset = ParticleDataset(self.options.TRAIN.datapath.eval, 
                                            data_type=self.options.TRAIN.datapath.eval_datatype,
                                            start=self.options.TRAIN.start_index,
                                            end=self.options.TRAIN.end_index,
                                            random_rot=False, window=3)
        self.test_dataset_length = len(self.test_dataset)
        
    
    def build_model(self):
        # gravity = [float(x) for x in self.options.gravity.split(',')]
        gravity = self.options.TRAIN.gravity
        print('---> set gravity', gravity)
        self.transition_model = ParticleNet(gravity=gravity).to(self.device)
        if self.options.TRAIN.pretrained != '':
            ckpt = torch.load(self.options.TRAIN.pretrained)
            if 'transition_model_state_dict' in ckpt:
                ckpt = ckpt['transition_model_state_dict']
            elif 'model_state_dict' in ckpt:
                ckpt = ckpt['model_state_dict']
            ckpt = {k:v for k,v in ckpt.items() if 'gravity' not in k}
            transition_model_state_dict = self.transition_model.state_dict()
            transition_model_state_dict.update(ckpt)
            self.transition_model.load_state_dict(transition_model_state_dict, strict=True)
            print(f'---> load pretrained transition model: {self.options.TRAIN.pretrained}')
    

    def get_learning_rate(self,optimizer):
        lr=[]
        for param_group in optimizer.param_groups:
            lr +=[ param_group['lr'] ]
        return lr
    

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=self.options.TRAIN.lr)


    def train(self):
        self.transition_model.train()
        global_step = self.start_step
        print(self.start_step, self.options.TRAIN.N_iters)
        for epoch_idx in range(self.start_step, self.options.TRAIN.N_iters):
            for _, data in tqdm(enumerate(self.dataloader), total=self.dataset_length,desc=f'Training ({epoch_idx+1}/{self.options.TRAIN.N_iters})'):
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}
                pos0 = data['particles_pos_0'][0]
                vel0 = data['particles_vel_0'][0]
                pos1 = data['particles_pos_1'][0]
                vel1 = data['particles_vel_1'][0]
                pos2 = data['particles_pos_2'][0]
                vel2 = data['particles_vel_2'][0]
                box = data['box'][0]
                box_normals = data['box_normals'][0]
                
                pred_pos_1, pred_vel_1, num_fluid_neighbors_1 = self.transition_model(pos0, vel0, box, box_normals)
                pred_pos_2, pred_vel_2, num_fluid_neighbors_2 = self.transition_model(pred_pos_1, pred_vel_1, box, box_normals)
                # calculate losses
                loss1 = self.weighted_mse_loss(pred_pos_1, pos1, num_fluid_neighbors_1)
                loss2 = self.weighted_mse_loss(pred_pos_2, pos2, num_fluid_neighbors_2)
                loss = 0.5 * loss1 + 0.5 * loss2
                
                # boundary loss
                bloss1 = self.cal_boundary_loss(pred_pos_1)
                bloss2 = self.cal_boundary_loss(pred_pos_2)
                loss = loss + bloss1 + bloss2

                if (global_step+1) % 10 == 0:
                    # record grad l2 norm to check grad_norm
                    grad_norms = self.cal_grad_norm(self.transition_model)
                    self.summary_writer.add_histogram('grad_l2norm_before', grad_norms, global_step)
                
                self.optimizer.zero_grad()
                loss.backward()
                if self.options.TRAIN.grad_clip_value != 0:
                    torch.nn.utils.clip_grad_norm_(self.transition_model.parameters(), self.options.TRAIN.grad_clip_value)
                self.optimizer.step()
                
                if (global_step+1) % self.options.TRAIN.log_interval == 0:
                    self.summary_writer.add_scalar('loss1', loss1.item(), global_step)
                    self.summary_writer.add_scalar('loss2', loss2.item(), global_step)
                    self.summary_writer.add_scalar('bloss1', bloss1.item(), global_step)
                    self.summary_writer.add_scalar('bloss2', bloss2.item(), global_step)
                    self.summary_writer.add_scalar('loss', loss.item(), global_step)
                    # record lr
                    lrs = self.get_learning_rate(self.optimizer)
                    self.summary_writer.add_scalar('lr', lrs[0], global_step)
                    # record grad l2 norm to check grad_norm
                    grad_norms = self.cal_grad_norm(self.transition_model)
                    self.summary_writer.add_histogram('grad_l2norm_after', grad_norms, global_step)

                if (global_step+1) % self.options.TRAIN.save_interval == 0:
                    trg_path = osp.join(self.modelpath, f'{global_step}.pt')
                    torch.save({'step':epoch_idx,
                                'model_state_dict':self.transition_model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()}, trg_path)
                    self.eval(global_step)
                global_step += 1
                    

    def eval(self, step_idx):
        self.transition_model.eval()
        self.eval_count += 1
        with torch.no_grad():
            dist_pred2gt_all = []
            self.fluid_error = FluidErrors()
            for data_idx in range(self.test_dataset_length):
                data = self.test_dataset[data_idx]
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in data.items()}
                
                box = data['box']
                box_normals = data['box_normals']
                if data_idx ==0:
                    pos_for_next_step, vel_for_next_step = data['particles_pos_0'],data['particles_vel_0']
                pred_pos, pred_vel, num_fluid_nn = self.transition_model(pos_for_next_step, vel_for_next_step, box, box_normals)
                pos_for_next_step, vel_for_next_step = pred_pos.clone(), pred_vel.clone()
                
                # evaluate transition model
                pos_t1 = data['particles_pos_1']
                vel_t1 = data['particles_vel_1']
                # eval pred2gt distance
                dist_pred2gt = self.fluid_error.cal_errors(pred_pos.cpu().numpy(), pos_t1.cpu().numpy(), data_idx+1)
                dist_pred2gt_all.append(dist_pred2gt)
                self.summary_writer.add_scalar(f'pred2gt_distance', dist_pred2gt, self.eval_count*self.test_dataset_length+data_idx+1)
                # save to obj
                if not osp.exists(osp.join(self.particlepath, f'{step_idx}')):
                    os.makedirs(osp.join(self.particlepath, f'{step_idx}'))
                particle_name = osp.join(self.particlepath, f'{step_idx}/pred_{data_idx+1}.obj')
                with open(particle_name, 'w') as fp:
                    record2obj(pred_pos, fp, color=[255, 0, 0]) # red
                # self.summary_writer.add_mesh(f'Mesh/pred_{data_idx+1}', vertices=pred_pos_1, colors=[255, 255, 0])
                particle_name = osp.join(self.particlepath, f'{step_idx}/gt_{data_idx+1}.obj')
                with open(particle_name, 'w') as fp:
                    record2obj(pos_t1, fp, color=[3, 168, 158])
            self.summary_writer.add_scalar('avg_pred2gt_distance', np.mean(dist_pred2gt_all), step_idx)
        self.transition_model.train()
        # print('Evaluation Done (on our data)')
        print('avg_pred2gt_distance at step {} is {}'.format(step_idx, np.mean(dist_pred2gt_all)))
        print()