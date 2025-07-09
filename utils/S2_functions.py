import math
import torch
import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from torch.autograd.functional import jvp

import utils.Lie as lie

dtype = torch.float


################################################################################################################
################################################################################################################


def get_jacobian(input_arg):
    n = input_arg.shape[0]
    if input_arg.shape == (n, 3):
        thetaphi = x_to_q(input_arg)
    elif input_arg.shape == (n, 2):
        thetaphi = input_arg
    else:
        print(input_arg.shape)
        return
    theta = thetaphi[:, 0].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    phi = thetaphi[:, 1].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    ct = torch.cos(theta)
    st = torch.sin(theta)
    cp = torch.cos(phi)
    sp = torch.sin(phi)
    J = torch.cat([torch.cat([ct * cp, -st * sp], dim=2),
                   torch.cat([ct * sp, st * cp], dim=2),
                   torch.cat([-st, torch.zeros(n, 1, 1).to(input_arg)], dim=2)],
                  dim=1)
    return J


def get_Riemannian_metric_S2(input_arg):
    n = input_arg.shape[0]
    if input_arg.shape == (n, 3):
        thetaphi = x_to_q(input_arg)
    elif input_arg.shape == (n, 2):
        thetaphi = input_arg
    else:
        print(input_arg.shape)
        return
    theta = thetaphi[:, 0].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    # phi = thetaphi[:, 1].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    ones = torch.ones(n, 1, 1).to(input_arg)
    zeros = torch.zeros(n, 1, 1).to(input_arg)
    st_2 = torch.sin(theta)**2
    G = torch.cat([
        torch.cat([ones, zeros], dim=2),
        torch.cat([zeros, st_2], dim=2)
    ], dim=1)
    return G


################################################################################################################
################################################################################################################


def q_to_x(thetaphi):
    if thetaphi.shape == (2,):
        theta = thetaphi[0]
        phi = thetaphi[1]
        ct = torch.cos(theta)
        st = torch.sin(theta)
        cp = torch.cos(phi)
        sp = torch.sin(phi)
        x = torch.tensor([st * cp, st * sp, ct]).to(thetaphi)
    else:
        theta = thetaphi[..., 0].unsqueeze(-1)
        phi = thetaphi[..., 1].unsqueeze(-1)
        ct = torch.cos(theta)
        st = torch.sin(theta)
        cp = torch.cos(phi)
        sp = torch.sin(phi)
        x = torch.cat([st * cp, st * sp, ct], dim=-1).to(thetaphi)
    return x.type(torch.DoubleTensor).to(thetaphi)


def qdot_to_xdot(qdot, q):
    if q.shape[-1] == 3:
        q = x_to_q(q)
    pf = get_jacobian(q)
    return (pf @ qdot.unsqueeze(-1)).squeeze(-1)


def xdot_to_qdot(xdot, q):
    eps = 1e-7
    if q.shape[-1] == 3:
        q = x_to_q(q)
    pf = get_jacobian(q)
    pf_t = pf.transpose(1, 2)
    pf_t_pf = pf_t  @ pf
    pf_t_pf = torch.clamp(pf_t_pf, min=eps)
    pf_t_pf_inv = torch.inverse(pf_t_pf)
    mult = pf_t_pf_inv @ pf_t
    return (mult @ xdot.unsqueeze(-1)).squeeze(-1)


def xdot_projection(xdot, x):
    if x.shape[-1] == 2:
        x = q_to_x(x)
    xcos = torch.einsum('ni, ni -> n', x, xdot).unsqueeze(-1)
    xdot_proj = xdot - xcos * x
    return xdot_proj


def x_to_q(x):
    eps = 1e-7
    if x.shape == (3,):
        ct = x[2]
        assert 1 > ct > -1, 'theta is equal to pi, infinite solution'
        theta = torch.acos(ct)
        st = torch.sqrt(1 - ct ** 2)
        st = torch.clip(st, min = eps, max=1-eps)
        sp = x[1] / st
        cp = x[0] / st
        if sp >= 0:
            phi = torch.acos(cp)
        else:
            phi = 2 * math.pi - torch.acos(cp)
        thetaphi = torch.tensor([theta, phi], dtype=dtype).to(x)
        

    else:
        n = x.shape[0]
        ct = torch.clip(x[:, 2].unsqueeze(-1), min = -1+eps, max=1-eps)
        assert (sum(sum(ct < -1)) + sum(sum(ct > 1))) == 0, 'theta is equal to pi, infinite solution'
        
        theta = torch.acos(ct)
        st = torch.sqrt(1 - ct ** 2)
        st = torch.clip(st, min = eps, max=1-eps)
        sp = torch.clip(x[:, 1].unsqueeze(-1) / st, min = -1+eps, max=1-eps)
        cp = torch.clip(x[:, 0].unsqueeze(-1) / st, min = -1+eps, max=1-eps)
        phi = torch.acos(cp)
        phi[sp < 0] = -phi[sp < 0] + 2 * math.pi
        thetaphi = torch.cat([theta, phi], dim=1)
        # breakpoint()
    return thetaphi


################################################################################################################
################################################################################################################


def exp_sphere(x, v):
    if x.shape[-1] == 2:
        x = q_to_x(x) # (n, 3)
    if len(v.shape) == 3:
        if len(x.shape) == 2:
            x = x.unsqueeze(-1) # (n, 1, 3)
    vnorm = torch.norm(v, dim=-1).unsqueeze(-1) # (n, 1) or (n, s, 1)
    x_new = torch.cos(vnorm) * x + torch.sin(vnorm) * (v / vnorm)
    return x_new


def gaussian_sampling(qtraj, std, batch_size):
    num_timesteps = qtraj.shape[0]
    X = q_to_x(qtraj)
    traj_samples = X[torch.randint(0, num_timesteps, [batch_size])].unsqueeze(-1).to(qtraj)
    Gaussian_v = torch.empty(batch_size, 2, 1, 1).normal_(mean=0, std=std).to(qtraj)
    e1_temp = torch.empty(batch_size, 3, 1).normal_(mean=0, std=1).to(qtraj)
    e1_temp2 = e1_temp - traj_samples @ (traj_samples.transpose(1, 2)) @ e1_temp
    e1_temp3 = torch.sqrt(torch.sum(e1_temp2 ** 2, 1)).unsqueeze(-1)
    e1 = e1_temp2 / e1_temp3
    e2 = torch.linalg.cross(traj_samples, e1, dim=-1)
    d_v = Gaussian_v[:, 0] * e1 + Gaussian_v[:, 1] * e2
    d_v_norm = torch.sqrt(torch.sum(d_v ** 2, 1)).unsqueeze(-1)
    Random_X = (traj_samples * torch.cos(d_v_norm) +
                d_v / d_v_norm * torch.sin(d_v_norm)).view(batch_size, 3)
    Random_q = x_to_q(Random_X)
    return Random_q.detach()


def tangent_gaussian_sampling(q, std, sample_size):
    if q.shape[-1] == 2:
        x = q_to_x(q)
    elif q.shape[-1] == 3:
        x = q
    if len(x.shape) == 1:
        squeezed = True
        x = x.unsqueeze(0)
    nx = len(x)
    x = x.unsqueeze(1) # n, 1, 3
    vsample_3d = torch.empty(nx, sample_size, 3).to(q).normal_(mean=0, std=std)
    aligned_norm = (torch.sum(x * vsample_3d, dim=-1)).unsqueeze(-1) # n, s, 1
    vsample_tangent = vsample_3d - (x * aligned_norm) # n, s, 3
    xsample = exp_sphere(x, vsample_tangent)
    if squeezed:
        xsample = xsample.squeeze(0)
    return xsample


def uniform_sampling(batch_size, return_local=True):
    eps = 0.001
    Random_ball = torch.from_numpy(np.random.normal(0, 1, [int(batch_size * 2), 3])).to(dtype)
    Random_ball_norm = torch.sqrt(torch.sum(Random_ball ** 2, dim=1).unsqueeze(-1))
    Random_ball = Random_ball[Random_ball_norm>eps][:batch_size]
    Random_ball_norm = Random_ball_norm[Random_ball_norm>eps][:batch_size]
    Random_X = Random_ball / Random_ball_norm
    if return_local:
        Random_q = x_to_q(Random_X)
        return Random_q.detach()
    else:
        return Random_X


def grid_sampling(batch):
    # canonical Fibonacci Lattice
    # source
    # https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    
    n = batch
    goldenRatio = (1 + 5**0.5)/2
    i = torch.arange(0, n)
    theta = 2 *torch.pi * i / goldenRatio
    phi = torch.arccos(1 - 2*(i+0.5)/n)
    x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
    grid = torch.cat([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)], dim=1) # [batch, 3]
    return grid


################################################################################################################
################################################################################################################


def vel_geo_sphere(x1, x2, t):
    # Might not be used..
    eps = 1e-7
    x1x2 = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    theta = torch.arccos(x1x2)  # (n, 1)
    term1 = -theta * torch.sin(theta * t) * x1
    
    temp1 = x2 - (x1x2 * x1)
    temp1_norm = torch.clamp(torch.norm(temp1, dim=1).unsqueeze(-1), min=eps)
    temp2 = temp1 / temp1_norm
    term2 = theta * torch.cos(theta * t) * temp2
    
    return term1 + term2


def vel_geo_0_sphere(x1, x2):
    eps = 1e-10
    x1x2 = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    x1x2 = torch.clip(x1x2, min= -1 + eps, max = 1 - eps)
    theta = torch.arccos(x1x2)  # (n, 1)
    temp1 = x2 - (x1x2 * x1)
    temp1_norm = torch.clamp(torch.norm(temp1, dim=1).unsqueeze(-1), min=eps)
    temp2 = temp1 / temp1_norm
    return theta * temp2


def get_closest_point_S2(x, traj, index=True):
    x_traj_dist = torch.einsum('ni, mi -> nm', x, traj)
    index_closest = torch.argmax(x_traj_dist, dim=1)
    if index:
        return traj[index_closest], index_closest
    else:
        return traj[index_closest]


def get_SO3_from_w_theta(w, theta):
    w = torch.tensor(w, dtype=torch.float).unsqueeze(0)
    wnorm = torch.norm(w)
    what = w / wnorm
    return lie.exp_so3(what * theta).squeeze(0).numpy()


def get_geodesic_dist_S2(x1, x2):
    eps = 1e-10
    x1x2 = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    theta = torch.arccos(torch.clip(x1x2, min=-1+eps, max=1-eps))  # (n, 1)
    return theta


def parallel_transport(x1, x2, V):
    eps = 1e-10
    x1x2_cross = torch.linalg.cross(x1, x2, dim=-1)
    x1x2_dot = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    x1x2_dot = torch.clip(x1x2_dot, min=-1 + eps, max= 1 - eps)
    theta = torch.arccos(x1x2_dot)
    w = x1x2_cross * theta
    Rot = lie.exp_so3(w)
    return torch.einsum('nij, nj -> ni', Rot, V)


def BCSDM_S2(xsample, eta, xtraj, xdottraj):
    eps = 1e-6
    if xsample.shape[-1] == 2:
        xsample = q_to_x(xsample)
    xtraj_closest, index_closest = get_closest_point_S2(xsample, xtraj, index=True)
    xdottraj_closest = xdottraj[index_closest]
    if eta < 1e30:
        V1 = parallel_transport(xtraj_closest, xsample, xdottraj_closest)
    if eta > 0:
        vel = vel_geo_0_sphere(xsample, xtraj_closest)
    # Only parallel transport case (eta = 0)
    if eta == 0: 
        return V1
    # Only contraction case
    if eta > 1e30:
        return vel
    
    if type(eta) != torch.Tensor:
        eta = torch.zeros(len(xsample), 1).to(xtraj) + eta
    elif len(eta.shape) == 1:
        eta = eta.unsqueeze(1).to(xtraj)
    return V1 + eta * vel


################################################################################################################
################################################################################################################


def xtraj_to_xdottraj(xtraj, dt):
    qtraj = x_to_q(xtraj)
    qdottraj = (qtraj[1:,:] - qtraj[:-1,:])/dt
    qdottraj = torch.cat([qdottraj, torch.zeros([1,qdottraj.shape[1]])], dim=0)
    xdottraj = qdot_to_xdot(qdottraj, qtraj)
    return xdottraj

def gen_traj_S2(x_input, model, eta=1, time_step=110, dt=0.03, model_type=None):
    # input : [batch, 3 or 2]
    if x_input.shape[-1] == 2:
        x_input = q_to_x(x_input)
    
    x_now  = x_input
    xtraj_list = [x_now.unsqueeze(1)]
    qtraj_list = [x_to_q(x_now).unsqueeze(1)]
    for j in range(time_step):
        q_dot = model(x_now.to(x_input), eta=eta)
        max_speed = 1
        q_dot = q_dot / torch.where(q_dot.norm(dim=1)>max_speed, q_dot.norm(dim=1)/max_speed, 1.).unsqueeze(1).repeat(1, q_dot.shape[1])
        q_now = x_to_q(x_now)
        q_next = q_now + dt * q_dot     # [batch, 2]
        x_now = q_to_x(q_next).to(torch.float) # [batch, 3]
        xtraj_list.append(x_now.unsqueeze(1).detach())
        qtraj_list.append(q_next.unsqueeze(1).detach())
    xtraj = torch.cat(xtraj_list, dim=1)
    qtraj = torch.cat(qtraj_list, dim=1)
    
    return xtraj, qtraj


def vec_3dim_to_2dim(x, vec):
    Jac = get_jacobian(x)
    return torch.einsum('nij, ni -> nj', Jac, vec)
