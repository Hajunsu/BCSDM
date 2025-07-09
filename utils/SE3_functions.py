import scipy
import torch
from utils import Lie as lie


################################################################################################################
################################################################################################################

def Rdot_to_wb(Rdot, R):
    wb = R.transpose(-1,-2) @ Rdot
    return wb


def wb_to_Rdot(wb, R):
    Rdot = R @ wb
    return Rdot


def Tdot_to_Vb(Tdot, T):
    if len(Tdot.shape) == 2 and Tdot.shape[-1] == 4:
        Tdot = Tdot.unsqueeze(0)
    if len(Tdot.shape) == 1 and Tdot.shape[-1] == 6:
        Tdot = Tdot.unsqueeze(0)
    if len(T.shape) == 2 and T.shape[-1] == 4:
        T = T.unsqueeze(0)
    if len(T.shape) == 1 and T.shape[-1] == 6:
        T = T.unsqueeze(0)
    
    Vb = lie.inverse_SE3(T) @ Tdot
    return Vb


def Tdot_to_qdot(Tdot, T):
    Vb = Tdot_to_Vb(Tdot, T)
    qdot = Vb_to_qdot(lie.screw_bracket(Vb).unsqueeze(-1), T)
    return qdot


def Tdot_to_wbpdot(Tdot, T):
    Vb = Tdot_to_Vb(Tdot, T)
    wb = lie.screw_bracket(Vb)[:,:3]
    pdot = Tdot[:, :3,3]
    return torch.cat([wb, pdot], dim=-1)


def Vb_to_qdot(Vb, T):
    R = T[:, :3, :3]
    Rt = R.transpose(1, 2)

    w = Vb[:, :3,:]
    v = R @ Vb[:, 3:,:]
    
    Dexp = lie.Dexp_so3(lie.skew(lie.log_SO3(R.to(T.device))))
    Temp = torch.einsum('nij, nkjl -> nkil', Rt, Dexp)
    Dexp_qdot_w = lie.skew(Temp.reshape(-1, 3, 3)).reshape(-1, 3, 3)

    qdot = (torch.cat([torch.inverse(Dexp_qdot_w.transpose(1, 2)) @ w, v], dim=1).squeeze(2))

    return qdot


def Vb_to_Tdot(Vb, T):
    if len(Vb.shape) == 2 and Vb.shape[-1] == 4:
        Vb = Vb.unsqueeze(0)
    if len(Vb.shape) == 1 and Vb.shape[-1] == 6:
        Vb = Vb.unsqueeze(0)
    if Vb.shape[-1] == 6:
        Vb = lie.skew_se3(Vb)
    if len(T.shape) == 2 and T.shape[-1] == 4:
        T = T.unsqueeze(0)
    if len(T.shape) == 1 and T.shape[-1] == 6:
        T = T.unsqueeze(0)
    
    Tdot = torch.einsum('nij, njk -> nik', T, Vb)
    return Tdot


def Vb_to_Vs(Vb, T):
    if Vb.shape[-1] == 6:
        Vb = lie.skew_se3(Vb)
    T_Vb = torch.einsum('nij, njk -> nik', T, Vb)
    Vs = torch.einsum('nij, njk -> nik', T_Vb, lie.inverse_SE3(T))
    return Vs


def wbpdot_to_Vb(wbpdot, T):
    R = T[:, :3, :3]
    w = wbpdot[:, :3]
    pdot = wbpdot[:, 3:].unsqueeze(-1)
    wb = lie.skew(w)
    Rdot = R @ wb
    Tdot = torch.cat([torch.cat([Rdot, pdot], dim=-1), torch.zeros(T.shape[0],1,4).to(T)], dim=-2)
    Vb = Tdot_to_Vb(Tdot, T)
    return Vb


def wbpdot_to_Tdot(wbpdot, T):
    Vb = wbpdot_to_Vb(wbpdot, T)
    Tdot = Vb_to_Tdot(Vb, T)
    return Tdot


def qdot_to_Vb(qdot, T):
    R = T[:, :3, :3]
    Rt = R.transpose(1, 2)
    
    qdot_w = qdot[:,:3,:].to(T)
    qdot_v = qdot[:,3:,:].to(T)
    
    Dexp = lie.Dexp_so3(lie.skew(lie.log_SO3(R)))
    Temp = torch.einsum('nij, nkjl -> nkil', Rt, Dexp)
    Dexp_qdot_w = lie.skew(Temp.reshape(-1, 3, 3)).reshape(-1, 3, 3)
    
    Vb = torch.cat([Dexp_qdot_w.transpose(1, 2) @ qdot_w ,Rt @ qdot_v], dim=1)
    
    return Vb


def Vs_to_Vb(Vs, T):
    if Vs.shape[-1] == 6:
        Vs = lie.skew_se3(Vs)
    Tinv_Vs = torch.einsum('nij, njk -> nik', lie.inverse_SE3(T), Vs)
    Vb = torch.einsum('nij, njk -> nik', Tinv_Vs, T)
    return Vb


################################################################################################################
################################################################################################################


def Gaussian_sampling_SE3(Ttraj, w_std, p_std, batch_size):
    eps = 1e-10
    if w_std == 0:
        w_std = eps
    if p_std == 0:
        p_std = eps
    
    T = Ttraj
    num_timesteps = T.shape[0]
    traj_samples = T[torch.randint(0, num_timesteps, (batch_size,))]
    w_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * w_std)
    p_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * p_std)
    
    Gaussian_w = w_distribution.sample((batch_size,)).to(Ttraj)
    Gaussian_p = p_distribution.sample((batch_size,)).unsqueeze(-1).to(Ttraj)
    R_samples = traj_samples[:, :3, :3] @ lie.exp_so3(Gaussian_w)
    p_samples = traj_samples[:, :3, 3:4] + Gaussian_p
    random_T = torch.cat([torch.cat([R_samples, p_samples], dim=2), 
                          torch.zeros(batch_size, 1, 4, device=Ttraj.device)],dim=1).detach()

    return random_T


def tangent_gaussian_sampling_SE3(q, std_R=0.2, std_p=1, sample_size=100):
    if q.shape[-1] == 6:
        if len(q.shape) == 1:
            squeezed = True
            x = lie.exp_so3_T(q.unsqueeze(0))
    elif q.shape[-2:] == (4, 4):
        x = q
        if len(x.shape) == 2:
            squeezed = True
            x = x.unsqueeze(0)
    nx = len(x)
    x = x.unsqueeze(1) # n, 1, 4, 4
    R = x[:, :, :3, :3]
    p = x[:, :, :3, 3].unsqueeze(-1)
    wsample = torch.empty(nx, sample_size, 3).to(q).normal_(mean=0, std=std_R)
    psample = torch.empty(nx, sample_size, 3, 1).to(q).normal_(mean=0, std=std_p)
    
    Rsample_at_I = lie.exp_so3(wsample.reshape(-1, 3)).reshape(nx, sample_size, 3, 3)
    Rsample = R @ Rsample_at_I
    xsample = torch.cat([Rsample, psample + p], dim=-1)
    last_row = torch.zeros(nx, sample_size, 1, 4).to(q)
    last_row[:, :, -1, -1] = 1 
    xsample = torch.cat([xsample, last_row], dim=-2)
    if squeezed:
        xsample = xsample.squeeze(0)
    return xsample


def SO3_uniform_sampling(batch_size):
    SO3_sampler = scipy.stats.special_ortho_group(dim=3)
    R = SO3_sampler.rvs(batch_size)
    return R


################################################################################################################
################################################################################################################


def geodesic_SO3(R1, R2, t): #t : 0 ~ 1
    if R1.shape[0] == 3 and R2.shape[0] != 3:
        R1 = R1.repeat(R2.shape[0],1,1)
    elif R1.shape[0] == 3:
        R1 = R1.unsqueeze(0)
    if R2.shape[0] == 3 and R1.shape[0] != 3:
        R2 = R2.repeat(R1.shape[0],1,1)
    elif R2.shape[0] == 3:
        R2 = R2.unsqueeze(0)
    
    geodesic_point = R1@lie.exp_so3(t*lie.log_SO3(torch.einsum('nji, njk -> nik', R1, R2))).squeeze()
    
    return geodesic_point.squeeze()


def get_geodesic_dist_SO3(R1, R2):
    if R1.shape[0] == 3 and R2.shape[0] != 3:
        R1 = R1.repeat(R2.shape[0],1,1)
    elif R1.shape[0] == 3:
        R1 = R1.unsqueeze(0)
    if R2.shape[0] == 3 and R1.shape[0] != 3:
        R2 = R2.repeat(R1.shape[0],1,1)
    elif R2.shape[0] == 3:
        R2 = R2.unsqueeze(0)
    
    dist = torch.linalg.matrix_norm(lie.log_SO3(torch.einsum('nji, njk -> nik', R1, R2)), dim=(1,2)).squeeze()
    
    return dist


def get_geodesic_dist_SE3(T1, T2, c=1, d=100):
    dist_R = get_geodesic_dist_SO3(T1[:,0:3,0:3], T2[:,0:3,0:3])
    dist_p = torch.norm(T1[:,0:3,3] - T2[:,0:3,3], dim=-1)
    dist = torch.sqrt(c*dist_R**2 + d*dist_p**2)
    return dist


def get_closest_point_SE3(Tsample, Ttraj, c=1, d=100, index=True):
    nb, _, _ = Tsample.shape
    nt, _, _ = Ttraj.shape
    Tsample = Tsample.unsqueeze(1).repeat(1, nt, 1, 1).reshape(nb*nt, 4, 4)
    Ttraj = Ttraj.unsqueeze(0).repeat(nb, 1, 1, 1).reshape(nb*nt, 4, 4)
    dist = get_geodesic_dist_SE3(Tsample, Ttraj, c=c, d=d).reshape(nb, nt)
    index_closest = torch.argmin(dist, dim=1)
    T_closest = Ttraj[index_closest,:,:]
    if index:
        return T_closest, index_closest
    else:
        return T_closest


def parallel_transport_SO3(R1, R2, V):
    w = lie.log_SO3(R1.transpose(-1,-2) @ R2)
    R1TV = R1.transpose(-1,-2) @ V
    V_parallel = R1 @ lie.exp_so3(0.5*w) @ R1TV @ lie.exp_so3(0.5*w)
    return V_parallel


def vel_geo_0_SO3(R1, R2): # vel at R1 to R2
    W = lie.log_SO3(torch.einsum('nji, njk -> nik', R1, R2.to(R1)))
    Rdot = torch.einsum('nij, njk -> nik', R1, W.to(R1))
    return Rdot


def BCSDM_SE3(Tsample, eta_R, eta_p, Ttraj, Tdottraj, c=1, d=100, version='Tdot'):
    if len(Tsample.shape) == 2:
        Tsample = Tsample.unsqueeze(0)
    
    T_closest, index_closest = get_closest_point_SE3(Tsample, Ttraj, c=c, d=d, index=True)
    
    # pdot
    pdot_parallel = Tdottraj[index_closest,0:3,3].to(Tsample)
    pdot_contract = T_closest[:,0:3,3].to(Tsample) - Tsample[:,0:3,3].to(Tsample)
    
    # Rdot
    Rdot_closest = Tdottraj[index_closest,0:3,0:3]
    Rdot_parallel = parallel_transport_SO3(Ttraj[index_closest,0:3,0:3], Tsample[:,0:3,0:3], Rdot_closest)
    Rdot_contract = vel_geo_0_SO3(Tsample[:,0:3,0:3], Ttraj[index_closest,0:3,0:3])
    
    if eta_R == torch.inf or eta_p == torch.inf:
        Tdot = torch.zeros_like(Tsample).to(Tsample)
        Tdot[:,:3,:3] = Rdot_contract
        Tdot[:,:3,3] = pdot_contract
    else:
        Tdot = torch.zeros_like(Tsample).to(Tsample)
        Tdot[:,:3,:3] = Rdot_parallel + eta_R*Rdot_contract
        Tdot[:,:3,3] = pdot_parallel + eta_p*pdot_contract
    if version == 'qdot':
        output = Tdot_to_qdot(Tdot, Tsample)
    elif version == 'wbpdot':
        output = Tdot_to_wbpdot(Tdot, Tsample)
    else:
        output = Tdot
    
    return output



################################################################################################################
################################################################################################################


def Rdot_projection(Rdot, R):
    return R@(R.permute(0,2,1)@Rdot - Rdot.permute(0,2,1)@R)/2


def Rtraj_to_Rdottraj(R):
    R1 = R[:-1].to(R)
    R2 = R[1:].to(R)
    W = lie.log_SO3(torch.einsum('nji, njk -> nik', R1, R2))
    Rdot = torch.einsum('nij, njk -> nik', R1, W.to(R))
    Rdot = torch.cat([Rdot, torch.zeros([1,3,3]).to(R)], dim=0)
    Rdot_proj = Rdot_projection(Rdot, R)
    return Rdot_proj


def Ttraj_to_Tdottraj(Ttraj, dt=0.002):
    Rtraj = Ttraj[:,0:3,0:3].type(torch.float).to(Ttraj)
    p = Ttraj[:,0:3,3].type(torch.float).to(Ttraj)
    Rdottraj = Rtraj_to_Rdottraj(Rtraj)
    pdottraj = torch.cat((p[1:]-p[:-1],torch.zeros([1,3]).to(Ttraj)),dim=0)
    Tdottraj = torch.zeros(Ttraj.shape).to(Ttraj)
    Tdottraj[:,0:3,0:3] = Rdottraj
    Tdottraj[:,0:3,3] = pdottraj
    return Tdottraj/dt


def Ttraj_to_Tdottraj_list(Ttraj_list, dt_list):
    Tdottraj_list = []
    for i in range(len(Ttraj_list)):
        Tdottraj = Ttraj_to_Tdottraj(Ttraj_list[i], dt_list[i])
        Tdottraj_list.append(Tdottraj)
    return Tdottraj_list


################################################################################################################
################################################################################################################


def SE3_update(T, Tdot, dt):
    wb = T[0:3,0:3].T @ Tdot[:3,:3]
    Rb = lie.exp_so3(wb.unsqueeze(0)*dt).squeeze()
    R = T[0:3,0:3] @ Rb
    p = Tdot[:3,3].to(T)*dt + T[:3,3]
    T_new = torch.eye(4,4)
    T_new[:3,:3] = R.squeeze()
    T_new[:3,3] = p
    return T_new.to(T)


def SE3_update_batch(T, Tdot, dt):
    wb = T[:,0:3,0:3].permute(0,2,1) @ Tdot[:,:3,:3]
    Rb = lie.exp_so3(wb*dt)
    R = T[:,0:3,0:3] @ Rb
    p = Tdot[:,:3,3].to(T)*dt + T[:,:3,3]
    T_new = torch.eye(4,4).unsqueeze(0).repeat(T.shape[0],1,1)
    T_new[:,:3,:3] = R
    T_new[:,:3,3] = p
    return T_new.to(T)


################################################################################################################
################################################################################################################


def mat2vec_SE3(X):    
    xshape = X.shape
    R = X[:, :3, :3]
    Rv = R.reshape(*(xshape[:-2]), -1)
    b = X[:, 0:3, 3]
    X_se3 = torch.cat((Rv, b), dim=-1)
    return X_se3


def mat2vec_SE3_batch(X):
    xshape = X.shape
    R = X[:, :, :3, :3]
    Rv = R.reshape(*(xshape[:-2]), -1)
    b = X[:, :, 0:3, 3]
    X_se3 = torch.cat((Rv, b), dim=-1)
    return X_se3


def vec2mat_SE3(V):
    xshape = V.shape
    R = V[:, :9].reshape(*(xshape[:-1]), 3, 3)
    b = V[:, 9:].unsqueeze(-1)
    X_se3 = torch.cat((R, b), dim=-1)
    eye = (torch.eye(4).to(V)[-1]).unsqueeze(0).unsqueeze(0).repeat(*xshape[:-1], 1, 1)
    X_se3 = torch.cat([X_se3, eye], dim=-2)
    return X_se3


def mat2vec_SE3_traj(Xtraj):
    n = Xtraj.shape[0]
    m = Xtraj.shape[1]
    R = Xtraj[:, :, :3, :3]
    Rv = R.reshape(n, m, -1)
    b = Xtraj[:, :, 0:3, 3]
    Xtraj_se3 = torch.cat((Rv, b), dim=2)
    return Xtraj_se3.reshape(n, -1)


def vec_6dim_to_12dim(x, vec):
    if x.shape[1:] == (12,):
        R = x[:, :9].reshape(-1, 3, 3)
    elif x.shape[1:] == (4, 4):
        R = x[:, :3, :3]

    w = vec[:, :3]
    v = vec[:, 3:]
    Rdot_I = lie.skew(w)
    Rdot_x = R @ Rdot_I
    vec_12dim = torch.cat([Rdot_x.reshape(-1, 9), v], dim=1)
    return vec_12dim


def vec_12dim_to_6dim(x, vec):
    if x.shape[1:] == (12,):
        R = x[:, :9].reshape(-1, 3, 3)
    elif x.shape[1:] == (4, 4):
        R = x[:, :3, :3]
    if vec.shape[1:] == (12,):
        Rdot = vec[:, :9].reshape(-1, 3, 3)
        pdot = vec[:, 9:]
    elif vec.shape[1:] == (4, 4):
        Rdot = vec[:, :3, :3]
        pdot = vec[:, :3, 3]
    W = R.transpose(1, 2) @ Rdot
    w = lie.skew(W)
    vec_6dim = torch.cat([w, pdot], dim=1)
    return vec_6dim