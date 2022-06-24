"""
Renderer
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .nerf import Embedding, NeRF
from utils.ray_utils import coarse_sample_ray
from utils.ray_utils import ImportanceSampling

from pytorch3d.ops import ball_query

class RenderNet(nn.Module):
    def __init__(self, cfg, near, far):
        super(RenderNet, self).__init__()
        self.cfg = cfg
        # self.ray_chunk_interpolate = self.cfg.ray_chunk_interpolate
        # self.ray_chunk = cfg.ray_chunk
        self.near = near
        self.far = far
        self.N_samples = cfg.ray.N_samples
        self.N_importance = cfg.ray.N_importance
        self.raduis = self.cfg.NN_search.search_raduis_scale * self.cfg.NN_search.particle_radius
        self.fix_radius = self.cfg.NN_search.fix_radius
        self.num_neighbor = self.cfg.NN_search.N_neighbor

        # build network
        self.embedding_xyz = Embedding(3, 10)
        in_channels_xyz = self.embedding_xyz.out_channels
        self.embedding_dir = Embedding(3, 4)
        in_channels_dir = self.embedding_dir.out_channels
        if cfg.encoding.density:
            self.embedding_density = Embedding(1, 4)
            in_channels_xyz += self.embedding_density.out_channels
        if cfg.encoding.var:
            in_channels_xyz += self.embedding_xyz.out_channels
        if cfg.encoding.smoothed_pos:
            in_channels_xyz += self.embedding_xyz.out_channels
        if cfg.encoding.smoothed_dir:
            in_channels_dir += self.embedding_dir.out_channels
        self.nerf_coarse = NeRF(in_channels_xyz=in_channels_xyz, in_channels_dir=in_channels_dir)
        self.nerf_fine = NeRF(in_channels_xyz=in_channels_xyz, in_channels_dir=in_channels_dir)


    def set_ro(self, cw):
        """
        get the camera position in world coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
        """
        ray_o = cw[:,3] # (3,)
        return ray_o

    
    def get_particles_direction(self, particles, ro):
        ros = ro.expand(particles.shape[0], -1)
        dirs = particles - ros
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        return dirs


    def mask_gather(self, points, idx):
        N, P, D = points.shape
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
        idx_expanded_mask = idx_expanded.eq(-1)
        idx_expanded = idx_expanded.clone()
        # Replace -1 values with 0 for gather
        idx_expanded[idx_expanded_mask] = 0
        # Gather points
        selected_points = points.gather(dim=1, index=idx_expanded)
        # Replace padded values
        selected_points[idx_expanded_mask] = 0.0
        return selected_points


    def _ball_query(self, query_points, points, radius, nsample):
        dists = torch.cdist(query_points, points) # N_ray, N_ray_points, N_phys_points
        dists_sorted, indices_sorted = torch.sort(dists, dim=-1)
        _dists, _indices = dists_sorted[:, :, :nsample], indices_sorted[:, :, :nsample]
        mask = (_dists > radius)
        _dists[mask] = 0.
        _indices[mask] = -1
        selected_points = self.mask_gather(points, _indices)
        return _dists, _indices, selected_points


    def get_search_raduis(self, R, z, f):
        dist = R * torch.abs(z) / f
        return dist.unsqueeze(-1)
    
    
    def smoothing_position(self, ray_pos, nn_poses, raduis, num_nn, exclude_ray=True, larger_alpha=0.9, smaller_alpha=0.1):
        dists = torch.norm(nn_poses - ray_pos.unsqueeze(-2), dim=-1)
        weights = torch.clamp(1 - (dists / raduis) ** 3, min=0)
        weighted_nn = (weights.unsqueeze(-1) * nn_poses).sum(-2) / (weights.sum(-1, keepdim=True)+1e-12)
        if exclude_ray:
            pos = weighted_nn
        else:
            if self.cfg.encoding.same_smooth_factor:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
            else:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
                alpha[num_nn.le(20)] = smaller_alpha
            pos = ray_pos * (1-alpha) + weighted_nn * alpha
        return pos, weights.sum(-1, keepdim=True)
    
    
    def search(self, ray_particles, particles, fix_radius):
        raw_data = particles.unsqueeze(0).repeat(ray_particles.shape[0], 1, 1)
        if fix_radius:
            radiis = self.raduis
            dists, indices, neighbors = ball_query(p1=ray_particles, 
                                                   p2=raw_data, 
                                                   radius=radiis, K=self.num_neighbor)
        # else:
        #     radiis = self.get_search_raduis(self.raduis, ray_particles[:,:,-1] - ro[-1], focal)
        #     dists, indices, neighbors = self._ball_query(ray_particles, raw_data, radiis, self.num_neighbor)
        return dists, indices, neighbors, radiis
    
        
    def embedding_local_geometry(self, dists, indices, neighbors, radius, ray_particles, rays, ro, sigma_only=False):
        """
        pos like feats
            1. smoothed positions
            2. ref hit pos, i.e., ray position
            3. density
            3. variance
        dir like feats
            1. hit direction, i.e., ray direction
            2. main direction after PCA
        """
        # calculate mask
        nn_mask = dists.ne(0)
        num_nn = nn_mask.sum(-1, keepdim=True)

        # hit pos and hit direction (basic in NeRF formulation)
        pos_like_feats = []
        hit_pos = ray_particles.reshape(-1,3)
        hit_pos_embedded = self.embedding_xyz(hit_pos)
        pos_like_feats.append(hit_pos_embedded)
        if not sigma_only:
            hit_dir = rays[:,3:]
            hit_dir_embedded = self.embedding_dir(hit_dir)
            hit_dir_embedded = torch.repeat_interleave(hit_dir_embedded, repeats=ray_particles.shape[1], dim=0)
            dir_like_feats = []
            dir_like_feats.append(hit_dir_embedded)
        # smoothing 
        smoothed_pos, density = self.smoothing_position(ray_particles, neighbors, radius, num_nn, exclude_ray=self.cfg.encoding.exclude_ray)
        smoothed_dir = self.get_particles_direction(smoothed_pos.reshape(-1, 3), ro)
        # density
        if self.cfg.encoding.density:
            density_embedded = self.embedding_density(density.reshape(-1, 1))
            pos_like_feats.append(density_embedded)
        # smoothed pos
        if self.cfg.encoding.smoothed_pos:
            smoothed_pos_embedded = self.embedding_xyz(smoothed_pos.reshape(-1, 3))
            pos_like_feats.append(smoothed_pos_embedded)
        # variance
        if self.cfg.encoding.var:
            vec_pp2rp = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            vec_pp2rp[nn_mask] = (neighbors - ray_particles.unsqueeze(-2))[nn_mask]
            vec_pp2rp_mean = vec_pp2rp.sum(-2) / (num_nn+1e-12)
            variance = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            variance[nn_mask] = ((vec_pp2rp - vec_pp2rp_mean.unsqueeze(-2))**2)[nn_mask]
            variance = variance.sum(-2) / (num_nn+1e-12)
            variance_embedded = self.embedding_xyz(variance.reshape(-1,3))
            pos_like_feats.append(variance_embedded)
        # smoothed dir
        if self.cfg.encoding.smoothed_dir:
            smoothed_dir_embedded = self.embedding_dir(smoothed_dir)
            dir_like_feats.append(smoothed_dir_embedded)
        if not sigma_only:
            return pos_like_feats, dir_like_feats, num_nn
        else:
            return pos_like_feats
        
    
    def render_image(self, rgbsigma, zvals, rays, noise_std, white_background):
        rgbs = rgbsigma[..., :3]
        sigmas = rgbsigma[..., 3]
        # convert these values using volume rendering (Section 4)
        deltas = zvals[:, 1:] - zvals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        
        deltas = deltas * torch.norm(rays[:,3:].unsqueeze(1), dim=-1)
        
        noise = 0.
        if noise_std > 0.:
            noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
        
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1)
        
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*zvals, -1) # (N_rays)
        
        if white_background:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)
        return rgb_final, depth_final, weights
    
    
    def forward(self, physical_particles, ro, rays, focal, c2w, use_disp=False, perturb=0, noise_std=0., white_background=True):
        """
        physical_particles: N_particles, 3
        ray_particles: N_ray, N_samples, 3
        zvals: N_rays, N_samples
        ro: 3, camera location
        rays: N_rays, 6
        """
        results = {}
        N_rays = rays.shape[0]
        # ---------------
        # coarse render
        # ---------------
        # coarsely sample
        z_values_0, ray_particles_0 = coarse_sample_ray(self.near, self.far, rays, self.N_samples, use_disp, perturb)
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(ray_particles_0, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0, dirs_like_feats_0, num_nn_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, ray_particles_0, rays, ro)
        input_feats_0 = torch.cat(pos_like_feats_0+dirs_like_feats_0, dim=1)
        # predict rgbsigma
        rgbsigma_0 = self.nerf_coarse(input_feats_0)
        mask_0 = torch.all(dists_0!=0, dim=-1, keepdim=True).float()
        if self.cfg.use_mask:
            rgbsigma_0 = rgbsigma_0.view(-1, self.N_samples, 4)*mask_0
        else:
            rgbsigma_0 = rgbsigma_0.view(-1, self.N_samples, 4)
        # render
        rgb_final_0, depth_final_0, weights_0 = self.render_image(rgbsigma_0, z_values_0, rays, noise_std, white_background)
        results['rgb0'] = rgb_final_0
        results['depth0'] = depth_final_0
        results['opacity0'] = weights_0.sum(1)
        results['num_nn_0'] = num_nn_0
        results['mask_0'] = mask_0.sum(1)
        
        # ---------------
        # fine render
        # ---------------
        if self.N_importance>0:
            ray_particles_1, z_values_1 = ImportanceSampling(z_values_0, weights_0, self.N_importance, rays[...,:3], rays[...,3:], det=(perturb==0))
            # search
            dists_1, indices_1, neighbors_1, radius_1 = self.search(ray_particles_1, physical_particles, self.fix_radius)
            # embedding attributes
            pos_like_feats_1, dirs_like_feats_1, num_nn_1 = self.embedding_local_geometry(dists_1, indices_1, neighbors_1, radius_1, ray_particles_1, rays, ro)
            input_feats_1 = torch.cat(pos_like_feats_1+dirs_like_feats_1, dim=1)
            # predict rgbsigma
            rgbsigma_1 = self.nerf_fine(input_feats_1)
            mask_1 = torch.all(dists_1!=0, dim=-1, keepdim=True).float()
            if self.cfg.use_mask:
                rgbsigma_1 = rgbsigma_1.view(-1, mask_1.shape[1], 4)*mask_1
            else:
                rgbsigma_1 = rgbsigma_1.view(-1, mask_1.shape[1], 4)
            # render
            rgb_final_1, depth_final_1, weights_1 = self.render_image(rgbsigma_1, z_values_1, rays, noise_std, white_background)
            results['rgb1'] = rgb_final_1
            results['depth1'] = depth_final_1
            results['opacity1'] = weights_1.sum(1)
            results['num_nn_1'] = num_nn_1
            results['mask_1'] = mask_1.sum(1)
        return results


    def coarse_rendering(self, physical_particles, ro, rays, focal, c2w, use_disp=False, perturb=0, noise_std=0., white_background=True):
        """
        physical_particles: N_particles, 3
        ray_particles: N_ray, N_samples, 3
        zvals: N_rays, N_samples
        ro: 3, camera location
        rays: N_rays, 6
        """
        results = {}
        N_rays = rays.shape[0]
        # ---------------
        # coarse render
        # ---------------
        # coarsely sample
        z_values_0, ray_particles_0 = coarse_sample_ray(self.near, self.far, rays, self.N_samples, use_disp, perturb)
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(ray_particles_0, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0, dirs_like_feats_0, num_nn_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, ray_particles_0, rays, ro)
        input_feats_0 = torch.cat(pos_like_feats_0+dirs_like_feats_0, dim=1)
        # predict rgbsigma
        rgbsigma_0 = self.nerf_coarse(input_feats_0)
        mask_0 = torch.all(dists_0!=0, dim=-1, keepdim=True).float()
        if self.cfg.use_mask:
            rgbsigma_0 = rgbsigma_0.view(-1, self.N_samples, 4)*mask_0
        else:
            rgbsigma_0 = rgbsigma_0.view(-1, self.N_samples, 4)
        # render
        rgb_final_0, depth_final_0, weights_0 = self.render_image(rgbsigma_0, z_values_0, rays, noise_std, white_background)
        results['rgb0'] = rgb_final_0
        results['depth0'] = depth_final_0
        results['opacity0'] = weights_0.sum(1)
        results['num_nn_0'] = num_nn_0
        results['mask_0'] = mask_0.sum(1)
        return results

    
    def fine_rendering(self, physical_particles, ro, rays, focal, c2w, use_disp=False, perturb=0, noise_std=0., white_background=True):
        results = {}
        # ---------------
        # coarse render
        # ---------------
        # coarsely sample
        z_values_0, ray_particles_0 = coarse_sample_ray(self.near, self.far, rays, self.N_samples, use_disp, perturb)
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(ray_particles_0, physical_particles, self.fix_radius)

        # --------
        # only need sigma
        pos_like_feats,= self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, ray_particles_0, rays, ro, sigma_only=True)
        input_feats = torch.cat(pos_like_feats, dim=1)
        sigmas_0 = self.nerf_coarse(input_feats, sigma_only=True)
        mask_0 = torch.all(dists_0!=0, dim=-1, keepdim=True).float()
        if self.cfg.use_mask:
            sigmas_0 = (sigmas_0.view(-1, self.N_samples, 1)*mask_0).squeeze(-1)
        else:
            sigmas_0 = (sigmas_0.view(-1, self.N_samples, 1)).squeeze(-1)
        # convert these values using volume rendering (Section 4)
        deltas = z_values_0[:, 1:] - z_values_0[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        deltas = deltas * torch.norm(rays[:,3:].unsqueeze(1), dim=-1)
        noise = 0.
        if noise_std > 0.:
            noise = torch.randn(sigmas_0.shape, device=sigmas_0.device) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas_0+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights_0 = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        
        # ---------------
        # fine render
        # ---------------
        assert self.N_importance>0
        if True:
            ray_particles_1, z_values_1 = ImportanceSampling(z_values_0, weights_0, self.N_importance, rays[...,:3], rays[...,3:], det=(perturb==0))
            # search
            dists_1, indices_1, neighbors_1, radius_1 = self.search(ray_particles_1, physical_particles, self.fix_radius)
            # embedding attributes
            pos_like_feats_1, dirs_like_feats_1, num_nn_1 = self.embedding_local_geometry(dists_1, indices_1, neighbors_1, radius_1, ray_particles_1, rays, ro)
            input_feats_1 = torch.cat(pos_like_feats_1+dirs_like_feats_1, dim=1)
            # predict rgbsigma
            rgbsigma_1 = self.nerf_fine(input_feats_1)
            mask_1 = torch.all(dists_1!=0, dim=-1, keepdim=True).float()
            if self.cfg.use_mask:
                rgbsigma_1 = rgbsigma_1.view(-1, mask_1.shape[1], 4)*mask_1
            else:
                rgbsigma_1 = rgbsigma_1.view(-1, mask_1.shape[1], 4)
            # render
            rgb_final_1, depth_final_1, weights_1 = self.render_image(rgbsigma_1, z_values_1, rays, noise_std, white_background)
            results['rgb1'] = rgb_final_1
            results['depth1'] = depth_final_1
            results['opacity1'] = weights_1.sum(1)
            results['num_nn_1'] = num_nn_1
            results['mask_1'] = mask_1.sum(1)
        return results
        
        