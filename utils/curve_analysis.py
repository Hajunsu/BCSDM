import torch
import numpy as np
import scipy.interpolate as interpolate

from utils import S2_functions as s2
from utils import SE3_functions as se3
from utils import Lie as lie

dtype = torch.float


def get_closest_dist_traj_S2(sample_traj, demo_traj):
    """calculate closest dist traj for lyapunov exponent

    Args:
        sample_traj (torh.tensor): (nb, nt, 2 or 3)
        demo_traj (torch.tensor): (nt, 2 or 3)

    Returns:
        torch.tensor : (nb, nt)
    """
    eps = 1e-10
    if sample_traj.shape[-1] == 2:
        sample_traj = s2.q_to_x(sample_traj)
    if demo_traj.shape[-1] == 2:
        demo_traj = s2.q_to_x(demo_traj)
    
    nb, nt, _ = sample_traj.shape
    sample_traj_flat = sample_traj.reshape(-1,3)
    
    demo_closest = s2.get_closest_point_S2(sample_traj_flat, demo_traj)
    dist_traj = s2.get_geodesic_dist_S2(sample_traj_flat, demo_closest[0])
    dist_traj = torch.clip(dist_traj, min=eps)
    return dist_traj.reshape(nb, nt)


def get_closest_dist_traj_SE3(Tsample_traj, Ttraj):
    eps = 1e-10
    nb, nt1, _, _ = Tsample_traj.shape
    nt2, _, _ = Ttraj.shape
    Tsample_traj = Tsample_traj.unsqueeze(2).repeat(1, 1, nt2, 1, 1)
    Ttraj = Ttraj.unsqueeze(0).unsqueeze(0).repeat(nb, nt1, 1, 1, 1)
    Tsample_traj = Tsample_traj.reshape(-1, 4, 4)
    Ttraj = Ttraj.reshape(-1, 4, 4)
    dist = se3.get_geodesic_dist_SE3(Tsample_traj, Ttraj, c=0, d=1)
    dist = dist.reshape(nb, nt1, nt2)
    dist_min, _ = torch.min(dist, dim=-1)
    return dist_min


def cal_lyapunov_exponent(distance_traj, terminal_time=1, eps=1e-6):
    # input = ((nd), nt)
    input_size = distance_traj.shape
    if len(input_size) != 2:
        distance_traj = distance_traj.unsqueeze(0)
    
    lamb_list = []
    for i in range(len(distance_traj)):
        for j in range(len(distance_traj[i])):
            if distance_traj[i][j] < eps:
                distance_truncated = distance_traj[i][:j+1]
                break
            else:
                distance_truncated = distance_traj[i]
        
        t_linspace = torch.linspace(0, terminal_time, len(distance_truncated)).to(distance_traj)
        log_dist_traj = torch.log(distance_truncated).unsqueeze(-1) # ((nd), nt, 1)
        log_d0 = log_dist_traj[..., 0:1, 0:1]
        lamb = (torch.pinverse(t_linspace.unsqueeze(-1)) 
                @ (log_d0 - log_dist_traj)).squeeze().squeeze()
        lamb_list.append(lamb)
    lamb = torch.tensor(lamb_list)
    return lamb


def cal_all_length_traj_S2(traj, num_delta_t=200_000):
    length_list = torch.zeros(num_delta_t)
    tck, u = interpolate.splprep([traj[:, 0].detach().numpy(), traj[:, 1].detach().numpy()], s=0)
    xi, yi = interpolate.splev(np.linspace(0, 1, num_delta_t + 1), tck)
    qi = torch.cat([
        torch.from_numpy(xi).to(torch.float).unsqueeze(1),
        torch.from_numpy(yi).to(torch.float).unsqueeze(1)
    ], dim=1).unsqueeze(-1)
    dqi = qi[1:] - qi[:-1]
    G_qi = s2.get_Riemannian_metric_S2(qi[:-1].squeeze())
    dxi = (G_qi @ dqi).squeeze(-1)
    dsi = torch.sqrt((dxi**2).sum(dim=1))
    length_list[0] = dsi[0]
    for i in range(1, num_delta_t):
        length_list[i] = length_list[i-1] + dsi[i]
    return qi.squeeze(), length_list
    

def convert_traj_to_unit_speed_S2(traj_local, num_delta_t=100_000, target_num_timestep=100, target_length=None, unit_length=True):
    eps = 1e-6
    for i in range(1, len(traj_local)):
        diff = (traj_local[i] - traj_local[i-1]).abs().sum()
        if diff  < eps:
            traj_local = traj_local[:i]
            break
    traj_dense, length_list_dense = cal_all_length_traj_S2(traj_local, num_delta_t=num_delta_t)
    total_length = length_list_dense[-1]
    if unit_length == True:
        total_time_length = 1.0
    else:
        total_time_length = total_length
    length_list_target = torch.linspace(0, total_length, target_num_timestep)
    traj_uniform = torch.zeros(target_num_timestep, 2)
    i = 0
    for j in range(num_delta_t):
        if i == target_num_timestep:
            break
        if length_list_dense[j] >= length_list_target[i]:
            traj_uniform[i] = traj_dense[j]
            i += 1
        else:
            pass
    traj_uniform[-1] = traj_dense[-1]
    return traj_uniform, total_time_length


class DistanceGaussianSamplerS2():
    def __init__(self, demo_traj, std_multiplier=0.3, num_component=5, uniform_speed=False):
        if demo_traj.shape[-1] == 3:
            demo_traj_global = demo_traj
            demo_traj_local = s2.x_to_q(demo_traj)
        elif demo_traj.shape[-1] == 2:
            demo_traj_global = s2.q_to_x(demo_traj)
            demo_traj_local = demo_traj
        else:
            raise ValueError("Demo traj shape error in 'distance_gaussian_sampling_sphere'.")
        if uniform_speed:
            demo_traj_local, ttl = convert_traj_to_unit_speed_S2(
                demo_traj_local, target_num_timestep=len(demo_traj_local)
                )
            demo_traj_global = s2.q_to_x(demo_traj_local)
        self.num_component = num_component
        skip_size = int(len(demo_traj_global) / (num_component))
        self.x_center = demo_traj_global[:num_component*skip_size:skip_size]
        gmm_portion_unnormalizaed = (np.linspace(1, 0, num_component+1)[:-1])
        self.gmm_portion = gmm_portion_unnormalizaed / gmm_portion_unnormalizaed.sum()
        self.gmm_threshold = np.zeros(num_component + 1)
        for i in range(num_component):
            self.gmm_threshold[i+1] = (self.gmm_portion[:i+1]).sum()
        self.std_list = gmm_portion_unnormalizaed * std_multiplier
    
    def choose_component(self, num_sample=100):
        uniform_sample = torch.rand(num_sample)
        n_comp_list = torch.zeros(self.num_component)
        for i in range(self.num_component):
            n_comp_list[i] = torch.sum(torch.ones_like(uniform_sample)[
                (uniform_sample >= self.gmm_threshold[i]) * 
                (uniform_sample < self.gmm_threshold[i+1])])
        return n_comp_list.to(torch.int)

    def sample(self, num_sample=100):
        n_comp_list = self.choose_component(num_sample=num_sample)
        sample_total = []
        for i in range(self.num_component):
            n_comp = n_comp_list[i]
            std = self.std_list[i]
            current_sample = s2.tangent_gaussian_sampling(
                                self.x_center[i], std=std, sample_size=n_comp)
            sample_total.append(current_sample)
        sample_total = torch.cat(sample_total, dim=0)
        return sample_total


class DistanceGaussianSamplerSE3():
    def __init__(self, demo_traj, std_multiplier_R=0.1, std_multiplier_p=0.3, num_component=5,):
        if demo_traj.shape[-1] == 6:
            demo_traj_global = lie.log_SO3_T(demo_traj)
            demo_traj_local = demo_traj
            self.mode = 'se3'
        elif demo_traj.shape[-2:] == (4, 4):
            demo_traj_global = demo_traj
            demo_traj_local = lie.screw_bracket(lie.log_SO3_T(demo_traj))
            self.mode = 'se3'
        else:
            raise ValueError("Demo traj shape error in 'distance_gaussian_sampling_SE3'.")
        self.num_component = num_component
        skip_size = int(len(demo_traj_global) / (num_component))
        self.x_center = demo_traj_global[:num_component*skip_size:skip_size]
        gmm_portion_unnormalizaed = (np.linspace(1, 0, num_component+1)[:-1])
        self.gmm_portion = gmm_portion_unnormalizaed / gmm_portion_unnormalizaed.sum()
        self.gmm_threshold = np.zeros(num_component + 1)
        for i in range(num_component):
            self.gmm_threshold[i+1] = (self.gmm_portion[:i+1]).sum()
        self.std_R_list = gmm_portion_unnormalizaed * std_multiplier_R
        self.std_p_list = gmm_portion_unnormalizaed * std_multiplier_p
    
    def choose_component(self, num_sample=100):
        uniform_sample = torch.rand(num_sample)
        n_comp_list = torch.zeros(self.num_component)
        for i in range(self.num_component):
            n_comp_list[i] = torch.sum(torch.ones_like(uniform_sample)[
                (uniform_sample >= self.gmm_threshold[i]) * 
                (uniform_sample < self.gmm_threshold[i+1])])
        return n_comp_list.to(torch.int)

    def sample(self, num_sample=100):
        n_comp_list = self.choose_component(num_sample=num_sample)
        sample_total = []
        for i in range(self.num_component):
            n_comp = n_comp_list[i]
            std_R = self.std_R_list[i]
            std_p = self.std_p_list[i]
            current_sample = se3.tangent_gaussian_sampling_SE3(
                self.x_center[i], std_R=std_R, std_p=std_p, sample_size=n_comp)
            sample_total.append(current_sample)
        sample_total = torch.cat(sample_total, dim=0)
        return sample_total