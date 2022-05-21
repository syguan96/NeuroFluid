"""
A model to driven particles.

I borrow code from `Lagrangian Fluid Simulation with Continuous Convolutions` to build the model to driven the particles.
"""

import torch
import torch.nn as nn
import open3d.ml.torch as ml3d
import numpy as np
import torch.nn.functional as F


class ParticleNet(nn.Module):
    def __init__(self,
                 kernel_size=[4, 4, 4],
                 radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True,
                 particle_radius=0.025,
                 timestep=1 / 50,
                 gravity=(0, -9.81, 0),
                 other_feats_channels=0,):
        super(ParticleNet, self).__init__()
        self.layer_channels = [32, 64, 64, 3]
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        
        # simulation parameters
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(6 * self.radius_scale * self.particle_radius)
        self.time_step = timestep
        self.register_buffer('gravity', torch.FloatTensor(gravity))

        self._all_convs = []

        # input layers
        self.conv0_fluid = self._conv_layer(name='conv0_fluid',
                                            in_channels=4+other_feats_channels,
                                            filters=self.layer_channels[0], 
                                            activation=None)
        self.conv0_obstacle = self._conv_layer(name='conv0_obstacle', 
                                                in_channels=3, 
                                                filters=self.layer_channels[0],
                                                activation=None)
        self.dense0_fluid = nn.Linear(in_features=4+other_feats_channels, out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid.weight)
        torch.nn.init.zeros_(self.dense0_fluid.bias)
    
        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i-1]
            if i == 1:
                in_ch *= 3 # three kinds of input
            out_ch = self.layer_channels[i]
            dense = nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(self.dense0_fluid.weight)
            torch.nn.init.zeros_(self.dense0_fluid.bias)
            setattr(self, f'dense{i}', dense)
            conv = self._conv_layer(name=f'conv{i}',
                                    in_channels=in_ch,
                                    filters=out_ch,
                                    activation=None)
            setattr(self, f'conv{i}', conv)
            self.denses.append(dense)
            self.convs.append(conv)
    
    def _window_poly6(self, R):
        """
        Poly6 kernel
        """
        return torch.clamp((1 - R)**3, 0, 1)

    def _conv_layer(self, name, in_channels, filters, activation=None):
        conv_fn = ml3d.layers.ContinuousConv

        window_fn = None
        if self.use_window:
            window_fn = self._window_poly6

        conv = conv_fn(kernel_size=self.kernel_size, 
                        activation=activation,
                        interpolation=self.interpolation,
                        coordinate_mapping=self.coordinate_mapping,
                        normalize=False, 
                        window_function=window_fn,
                        radius_search_ignore_query_points=True,
                        in_channels=in_channels,
                        filters=filters
                        )
        
        self._all_convs.append((name, conv))
        return conv

    def integrate_pos_vel(self, pos, vel):
        dt = self.time_step
        vel_new = vel + self.gravity * dt
        pos_new = pos + (vel + vel_new) / 2 * dt
        return pos_new, vel_new

    def compute_pose_correction(self, pos_new, vel_new, other_feats, box, box_feats, fixed_radius_search_hash_table):
        """
        Core of this network
        """
        filter_extent = torch.tensor(self.filter_extent)
        fluid_feats = [torch.ones_like(pos_new[:, 0:1]), vel_new]
        if other_feats is not None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, axis=-1)

        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos_new, pos_new, filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos_new, filter_extent)

        feats = torch.cat([self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid], axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = F.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos_new, pos_new, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv0_fluid.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
        return self.pos_correction

    def update_pos_vel(self, pos, pos_new, pos_delta):
        dt = self.time_step
        pos_new_corrected = pos_new + pos_delta
        vel_new_corrected = (pos_new_corrected - pos) / dt
        return pos_new_corrected, vel_new_corrected


    def forward(self, pos, vel, box, box_feats, feats=None, fixed_radius_search_hash_table=None):
        # pos, vel, feats, box, box_feats = inputs
        
        # first apply gravity
        pos_new, vel_new = self.integrate_pos_vel(pos, vel)

        # calculate position deltas
        pos_deltas = self.compute_pose_correction(pos_new, vel_new, feats, box, box_feats, fixed_radius_search_hash_table)

        # correct the pos and vel
        pos_new_corrected, vel_new_corrected = self.update_pos_vel(pos, pos_new, pos_deltas)

        return pos_new_corrected, vel_new_corrected, self.num_fluid_neighbors


if __name__ == '__main__':
    particlenet = ParticleNet()
    print('---- Passed ----')
