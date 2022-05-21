import torch
from kornia import create_meshgrid
from torch.functional import norm

def get_dist_particle2ray(p, ray_o, ray_direction):
    """
    given a point whose position is p, return it's distance from the ray
    pytorch tensor
    """
    vec_r2p = p - ray_o
    # import ipdb;ipdb.set_trace()
    assert torch.norm(ray_direction)-1 < 1e-3, 'make sure direction is normalized, in fact:{}'.format(torch.norm(ray_direction))
    t = torch.dot(ray_direction, vec_r2p)
    q = ray_o + t * ray_direction
    vec_q2p = p - q
    dist = torch.linalg.norm(vec_q2p, dim=-1)
    return dist

def get_dist_particle2ray_batch(particle, ray_o, ray_direction):
    """
    Input:
        particle: particle position, (N, 1, 3)
        ray_o: original point of rays, (N, chunk, 3)
        ray_direction: direction of rays, (N, chunk, 3)
    Reutrn:
        dists: (N, chunk)
    """
    # import ipdb; ipdb.set_trace()
    vec_r2p = particle - ray_o # N,chunk,3
    t = (ray_direction * vec_r2p).sum(-1).unsqueeze(-1)
    qq = ray_o + t * ray_direction # N,chunk,3
    vec_q2p = particle - qq
    dist = torch.linalg.norm(vec_q2p, dim=-1) # N,chunk
    return dist

def assign_paticles_to_rays(particles, rays, chunk, particle_chunk=0, select_ray=1, thres=1e10):
    """
    assign particles to the nearest ray
    particles: (num_particles, 3)
    rays: (num_ray, 6)
        [...,:3] is the origin point
        [...,3:] is the normalized direction
    """
    num_particles = particles.shape[0]
    num_ray = rays.shape[0]

    # build a struct to save ray and it's particles
    struct_ray = {}

    for p_i in range(0, num_particles, particle_chunk):
        particle_batch = particles[p_i: p_i+particle_chunk]
        dists = []
        # calculate the distance to every ray
        for r_i in range(0, num_ray, chunk):
            ray_batch = rays[r_i:r_i+chunk]
            ray_o_batch = ray_batch[:, :3]
            ray_dir_batch = ray_batch[:, 3:]
            dist_batch = get_dist_particle2ray_batch(particle_batch.unsqueeze(1),
                                                     ray_o_batch.unsqueeze(0).repeat(particle_batch.shape[0], 1, 1),
                                                     ray_dir_batch.unsqueeze(0).repeat(particle_batch.shape[0], 1, 1)) # (particle_chunk, ray_chunk)
            dists.append(dist_batch)
        dists = torch.cat(dists, 1)#.cpu() # (particle_chunk, N_rays)
        # import ipdb;ipdb.set_trace()
        # print(p_i)
        assert select_ray > 0
        indices_chosed_ray = torch.argsort(dists, dim=1)[:, :select_ray] # (particle_chunk, num_selected_ray)
        for p_idx in range(indices_chosed_ray.shape[0]):
            particle = particle_batch[p_idx]
            for r_idx in range(select_ray):
                index_ray = indices_chosed_ray[p_idx, r_idx].item()
                chosed_ray = rays[index_ray]
                # import ipdb;ipdb.set_trace()
                if index_ray not in struct_ray:
                    # print(index_nearest_ray)
                    struct_ray[index_ray] = {'ray': chosed_ray, 'particles': [particle]}
                else:
                    struct_ray[index_ray]['particles'].append(particle)

    # todo: what if no particles in some ray?
    for k, v in struct_ray.items():
        v['particles'] = torch.stack(v['particles'])
    return struct_ray


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    # rays_d = rays_d.view(-1, 3)
    # rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples

def ImportanceSampling(zvals, weights, N_importance, rays_o, rays_d,det):
    zvals_mid = 0.5 * (zvals[...,1:] + zvals[..., :-1])
    z_samples = sample_pdf(zvals_mid, weights[:,1:-1], N_importance, det).detach()
    z_samples, _ = torch.sort(torch.cat([zvals, z_samples], -1), -1)
    
    xyz = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None]
    
    return xyz, z_samples


def coarse_sample_ray(near, far, rays, N_samples, use_disp, perturb):
    """
    coarsely sample N_sample points in each ray
    """
    z_steps = torch.linspace(0,1,N_samples,device=rays.device)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)
    
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    N_rays = rays.shape[0]
    z_vals = z_vals.expand(N_rays, N_samples)
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand
    
    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
    return z_vals, xyz_coarse_sampled