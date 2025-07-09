import os
import time
import tqdm
import scipy
import torch
import platform
import numpy as np
from functools import partial

from models import load_pretrained
import utils.S2_functions as s2
import utils.S2_vis as vis_s2
import utils.curve_analysis as curve


class S2_metric():
    def __init__(self, root=None, identifier=None, config_file=None, ckpt_file=None, demo_length=None,
                 data_path= 'datasets/S2_demos.pt', model_type = 'bc-deepovec',
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),):
        
        # set device
        self.device = device
        
        # set demo
        self.demo_length = demo_length
        S2_demos = torch.load(data_path)
        self.xtraj_origin = []
        self.xdottraj_origin = []
        self.qtraj_origin = []
        for data_num in range(len(S2_demos)):
            self.xtraj_origin.append(S2_demos[data_num]['xtraj'].unsqueeze(0).to(torch.float).to(self.device))
            self.xdottraj_origin.append(S2_demos[data_num]['xdottraj'].unsqueeze(0).to(torch.float).to(self.device))
            self.qtraj_origin.append(s2.x_to_q(S2_demos[data_num]['xtraj']).unsqueeze(0).to(torch.float).to(self.device))
        self.xtraj_origin = torch.cat(self.xtraj_origin, dim=0)
        self.xdottraj_origin = torch.cat(self.xdottraj_origin, dim=0)
        self.qtraj_origin = torch.cat(self.qtraj_origin, dim=0)
        
        # set model
        self.model_type = model_type
        model, _ = load_pretrained(identifier, config_file, ckpt_file, root)
        self.model = model.to(self.device)
        
        # set vectors and params
        self.params = sum(p.numel() for p in self.model.deeponet_parallel.trunck_net.parameters()) + sum(p.numel() for p in self.model.deeponet_contract.trunck_net.parameters())
        
        # set demo_short for evaluation
        if demo_length is not None:
            self.xtraj = []
            self.xdottraj = []
            self.qtraj = []
            step = int(len(S2_demos[0]['xtraj']) / demo_length)
            for data_num in range(len(S2_demos)):
                xtraj_short = torch.flip(S2_demos[data_num]['xtraj'], [0])
                xdottraj_short = torch.flip(S2_demos[data_num]['xdottraj'], [0])
                qtraj_short = torch.flip(s2.x_to_q(S2_demos[data_num]['xtraj']), [0])
                xtraj_short = torch.flip(xtraj_short[0:-1:step,:], [0])
                xdottraj_short = torch.flip(xdottraj_short[0:-1:step,:], [0])
                qtraj_short = torch.flip(qtraj_short[0:-1:step,:], [0])
                self.xtraj.append(xtraj_short.unsqueeze(0).to(torch.float).to(self.device))
                self.xdottraj.append(xdottraj_short.unsqueeze(0).to(torch.float).to(self.device))
                self.qtraj.append(qtraj_short.unsqueeze(0).to(torch.float).to(self.device))
            self.xtraj = torch.cat(self.xtraj, dim=0)
            self.xdottraj = torch.cat(self.xdottraj, dim=0)
            self.qtraj = torch.cat(self.qtraj, dim=0)
        
        # set samples
        self.xsamples = None
        self.qsamples = None
        self.xsample_trajs = [[]]*len(self.xtraj)
        self.qsample_trajs = [[]]*len(self.xtraj)
        
        self.q_trans = torch.tensor([0., 0.5]).to(self.device)
    
    def model_forward(self, x_input, data_num, eta=1, device=None, vis=False): # n-dimensional input
        if device is None:
            device = self.device
        xtraj_input = self.xtraj_origin[data_num].unsqueeze(0).repeat(len(x_input),1,1).to(device)
        output = self.model.to(device)(x=x_input.to(device), xtraj=xtraj_input, eta=eta)
        if output.shape[-1] == 3:
            output = s2.xdot_to_qdot(output, q=s2.x_to_q(x_input))
        
        return output
    
    def sampling_traj(self, batch_size=100, std=0.1, type='gaussian'):
        xsamples_list = []
        qsamples_list = []
        for i in range(len(self.xtraj)):
            if type == 'gaussian':
                qstart = s2.x_to_q(self.xtraj[i][0]).unsqueeze(0).to(self.device)
                qsamples = s2.gaussian_sampling(qstart, std=std, batch_size=batch_size)
                xsamples = s2.q_to_x(qsamples)
            elif type == 'distance':
                dd = curve.DistanceGaussianSamplerS2(self.qtraj[i], num_component=100, std_multiplier=std)
                xsamples = dd.sample(batch_size)
                qsamples = s2.x_to_q(xsamples)
            xsamples_list.append(xsamples.unsqueeze(0))
            qsamples_list.append(qsamples.unsqueeze(0))
        self.xsamples = torch.cat(xsamples_list, dim=0).to(self.device)
        self.qsamples = torch.cat(qsamples_list, dim=0).to(self.device)
    
    def save_samples(self, file_root, file_name):
        path = os.path.join(file_root, file_name)
        samples = {'xsamples' : np.array(self.xsamples.cpu()), 'qsamples' : np.array(self.qsamples.cpu())}
        scipy.io.savemat(path, samples)
    
    def load_samples(self, file_root, file_name):
        path = os.path.join(file_root, file_name)
        samples = scipy.io.loadmat(path)
        self.xsamples = torch.tensor(samples['xsamples']).to(self.device)
        self.qsamples = torch.tensor(samples['qsamples']).to(self.device)
    
    def generate_sample_trajectory(self, data_num=None, eta=1, time_step=110, dt=0.03):  
        if data_num is None:
            data_range = range(len(self.xtraj))
        elif type(data_num) is list:
            data_range = data_num
        elif type(data_num) is int:
            data_range = [data_num]
        else:
            raise Exception("Invalid data_num type!")
        
        for data_num in tqdm.tqdm(data_range, desc='Generating sample trajs'):
            vf = partial(self.model_forward, data_num=data_num, device=self.device)
            xtraj, qtraj = s2.gen_traj_S2(self.xsamples[data_num], vf, eta, time_step, dt, self.model_type)
            self.xsample_trajs[data_num] = xtraj
            self.qsample_trajs[data_num] = qtraj
    
    def fit_traj_error(self, data_num=None, eta=1):
        if data_num is None:
            data_range = range(len(self.xtraj))
        else:
            data_range = [data_num]
        error_list = []
        for data_num in data_range:
            model_out = self.model_forward(data_num=data_num, x_input=self.xtraj[data_num], eta=eta)
            model_out_xdot = s2.qdot_to_xdot(model_out, self.qtraj[data_num])
            error = ((self.xdottraj[data_num]-model_out_xdot)**2).sum(dim=1).sqrt().mean()
            xdot_max = torch.max((self.xdottraj_origin[data_num]**2).sum(dim=1).sqrt())
            
            error_list.append(error / xdot_max)
        
        total_error_std, total_error_mean = torch.std_mean(torch.tensor(error_list))
        return total_error_std.detach().numpy().tolist(), total_error_mean.detach().numpy().tolist()
    
    def fit_all_error(self, data_num=None, batch=1000, eta=1):
        if data_num is None:
            data_range = range(len(self.xtraj))
        else:
            data_range = [data_num]
        
        grid_sphere = s2.grid_sampling(batch=batch).to(self.device)
        error_list = []
        for d_num in data_range:
            qdot_output = self.model_forward(grid_sphere, d_num, eta)
            if torch.sum(torch.isnan(qdot_output))>0:
                print('nan data_num :', d_num)
                print('nan qdot out :', qdot_output)  
            xdot_gvf = s2.BCSDM_S2(grid_sphere, eta, self.xtraj_origin[d_num], self.xdottraj_origin[d_num])
            xdot_output = s2.qdot_to_xdot(qdot_output, s2.x_to_q(grid_sphere))
            error = ((xdot_output - xdot_gvf)**2).sum(dim=1).sqrt().mean()
            error_list.append(error)
        total_error_std, total_error_mean = torch.std_mean(torch.tensor(error_list))
        return total_error_std.detach().numpy().tolist(), total_error_mean.detach().numpy().tolist()
    
    
    def mimic_error(self, data_num, eta, mimicking):
        if data_num is None:
            data_range = range(len(self.xtraj))
        else:
            data_range = [data_num]
        
        xsample_list = []
        for i in data_range:
            xsample_list.append(self.xsamples[i])
        
        parallel_error_list = []
        for d_num in tqdm.tqdm(data_range, desc='Calculating parallel errors'):
            xdottraj_parallel = s2.BCSDM_S2(xsample_list[d_num], 0,
                                                    self.xtraj_origin[d_num],
                                                    self.xdottraj_origin[d_num])
            qdottraj_output = self.model_forward(xsample_list[d_num],
                                          d_num, eta, device=self.device)
            xdottraj_output = s2.qdot_to_xdot(qdottraj_output, xsample_list[d_num])
            xdot_max = torch.max((self.xdottraj_origin[d_num]**2).sum(dim=1).sqrt())
            
            if mimicking.error == 'real':
                parallel_error = ((xdottraj_parallel - xdottraj_output)**2).sum(dim=1).sqrt().mean()
                parallel_error_list.append(parallel_error / xdot_max)
            elif mimicking.error == 'direction':
                parallel_vel = (xdottraj_parallel**2).sum(dim=1).sqrt().unsqueeze(-1).repeat(1,3)
                output_vel = (xdottraj_output**2).sum(dim=1).sqrt().unsqueeze(-1).repeat(1,3)
                parallel_vel[parallel_vel==0] = 1.0
                output_vel[output_vel==0] = 1.0
                parallel_error = ((xdottraj_parallel/parallel_vel - xdottraj_output/output_vel)**2).sum(dim=1).sqrt().mean()
                parallel_error_list.append(parallel_error)
        
        parallel_error_std, parallel_error_mean = torch.std_mean(torch.tensor(parallel_error_list))
        
        return parallel_error_std.detach().numpy().tolist(), parallel_error_mean.detach().numpy().tolist()
    
            
    def lyapunov_exp(self, data_num, eps):
        if data_num is None:
            data_range = range(len(self.xtraj))
        else:
            data_range = [data_num]
        
        ## closest point version ##
        lamb_list = []
        for d_num in data_range:
            dist_traj = curve.get_closest_dist_traj_S2(self.xsample_trajs[d_num], self.xtraj[d_num])
            lamb = curve.cal_lyapunov_exponent(dist_traj, eps=eps) 
            lamb_list.append(lamb.unsqueeze(0))
        
        lamb = torch.cat(lamb_list, dim=0)
        lamb_std, lamb_mean = torch.std_mean(lamb.mean(dim=-1))
        return lamb_std.detach().numpy().tolist(), lamb_mean.detach().numpy().tolist()
    
    
    def vis(self, data_num, eta, plane, sphere, file_name=None):
        if data_num is None:
            data_range = range(len(self.xtraj))
        else:
            data_range = [data_num]
            
        for data_num in tqdm.tqdm(data_range, desc='Plotting'):
            file_name_vis = file_name + str(data_num)
            vf = partial(self.model_forward, data_num=data_num, device='cpu', vis=True)
            qtraj_vis = self.qtraj[data_num]
            
            if plane or sphere:
                # nq : windows -> 100 / linux -> 101
                if platform.system() == 'Windows':
                    nq = 100
                else:
                    nq = 101
                
                res_local = vis_s2.streamline_local(qtraj_vis.detach().cpu(), vf, eta = eta, rot_view=-1.5, w=[0,0,1], nq=nq,
                                                    for_sphere=True, for_wandb=False, for_metric=True, sample_trajs=None,
                                                    vector=True, file_name=file_name_vis)
            
            if sphere:
                vis_s2.streamline_sphere(qtraj_vis.detach().cpu(), rot_view=-1.5, res=res_local,
                                         for_wandb=True, sample_trajs=None, vector=True, file_name=file_name_vis)
    
    
    def cvf_mvf_error(self, data_num):
        if data_num is None:
            data_range = range(len(self.xtraj))
        else:
            data_range = [data_num]
        
        grid_sphere = s2.grid_sampling(batch=1000).to(self.device)
        
        mvf_error_list = []
        cvf_error_list = []
        for d_num in data_range:
            xdot_parallel = s2.BCSDM_S2(grid_sphere, 0,
                                                    self.xtraj_origin[d_num],
                                                    self.xdottraj_origin[d_num])
            xdot_contract = s2.BCSDM_S2(grid_sphere, torch.inf,
                                                    self.xtraj_origin[d_num],
                                                    self.xdottraj_origin[d_num])
            
            
            qdot_parallel_out = self.model_forward(grid_sphere, d_num, 0)
            qdot_contract_out = self.model_forward(grid_sphere, d_num, torch.inf)
            
            xdot_parallel_out = s2.qdot_to_xdot(qdot_parallel_out, grid_sphere)
            xdot_contract_out = s2.qdot_to_xdot(qdot_contract_out, grid_sphere)
                        
            mvf_error = ((xdot_parallel - xdot_parallel_out)**2).sum(dim=1).sqrt().mean()
            cvf_error = ((xdot_contract - xdot_contract_out)**2).sum(dim=1).sqrt().mean()
            mvf_error_list.append(mvf_error)
            cvf_error_list.append(cvf_error)
        
        mvf_error_std, mvf_error_mean = torch.std_mean(torch.tensor(mvf_error_list))
        cvf_error_std, cvf_error_mean = torch.std_mean(torch.tensor(cvf_error_list))
        
        mvf_std = mvf_error_std.detach().numpy().tolist()
        mvf_mean = mvf_error_mean.detach().numpy().tolist()
        cvf_std = cvf_error_std.detach().numpy().tolist()
        cvf_mean = cvf_error_mean.detach().numpy().tolist()
        
        return cvf_std, cvf_mean, mvf_std, mvf_mean 