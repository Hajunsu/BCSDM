import os
import tqdm
import scipy
import torch
import pickle
import numpy as np
from models import load_pretrained

import utils.curve_analysis as curve
import utils.SE3_functions as se3
import utils.Lie as lie


class SE3_metric():
    def __init__(self, root=None, identifier=None, config_file=None, ckpt_file=None,
                 data_path='datasets/SE3_demos.pt',
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 model_type = 'bc-deepovec'):
        
        # set device
        self.device = device
        self.model_type = model_type
        self.load_demo(data_path)

        # load model
        self.model = load_pretrained(identifier, config_file, ckpt_file, root)[0].to(self.device)
        self.params = self.params = sum(p.numel() for p in self.model.deeponet_parallel.trunck_net.parameters()) + sum(p.numel() for p in self.model.deeponet_contract.trunck_net.parameters())

    def load_demo(self, data_path):
        self.Ttraj_list = []
        self.Tdottraj_list = []
        self.dt_list = []

        SE3_demos = torch.load(data_path, weights_only=False)
        for data_num in range(len(SE3_demos)):
            self.Ttraj_list.append(torch.tensor(SE3_demos[data_num]['Ttraj']).unsqueeze(0))
            self.Tdottraj_list.append(torch.tensor(SE3_demos[data_num]['Tdottraj']).unsqueeze(0))
            self.dt_list.append(torch.tensor(SE3_demos[data_num]['dt']).unsqueeze(0))
                
        self.Ttrajs = torch.cat(self.Ttraj_list, dim=0).type(torch.float32).to(self.device)
        self.Tdottrajs = torch.cat(self.Tdottraj_list, dim=0).type(torch.float32).to(self.device)
        self.dts = torch.cat(self.dt_list, dim=0).type(torch.float32).to(self.device)
        
        self.wbpdot_max_list = []
        for d_num in range(len(self.Ttrajs)):
            wbpdot_traj = se3.Tdot_to_wbpdot(self.Tdottrajs[d_num], self.Ttrajs[d_num])
            wbpdot_max = wbpdot_traj.norm(dim=-1).max()
            self.wbpdot_max_list.append(wbpdot_max)
        
    def model_forward(self, T_input, data_num, eta_R, eta_p, device=None, output_type='Tdot'):
        if device is None:
            device = self.device
        
        T_input = T_input.to(self.device).type(torch.float32)
        if len(T_input.shape) == 2:
            T_input = T_input.unsqueeze(0)
        Ttraj_input = self.Ttraj_list[data_num].repeat(T_input.shape[0],1,1,1)
        model_output = self.model.forward(T_input, Ttraj_input, eta_R, eta_p)
        if model_output.shape[-1] == 6:
            model_output = se3.wbpdot_to_Tdot(model_output, T_input)
        
        if output_type == 'wbpdot':
            wbpdot_output = se3.Tdot_to_wbpdot(model_output, T_input)
            wbpdot_max = self.wbpdot_max_list[data_num]
            
            wbpdot_output_norm = wbpdot_output.norm(dim=-1).clip(min=1e-7)
            wbpdot_output_norm_final = wbpdot_output_norm.clip(max=wbpdot_max*2)
            wbpdot_output_norm = wbpdot_output_norm.unsqueeze(-1).repeat(1,6)
            wbpdot_output_norm_final = wbpdot_output_norm_final.unsqueeze(-1).repeat(1,6)
            model_output = wbpdot_output / wbpdot_output_norm * wbpdot_output_norm_final

        output = model_output.squeeze()
        if torch.isnan(output).int().sum() > 0:
            print("nan in model forward")
        
        return output
    
    def sampling_traj(self, batch_size=100, w_std=0.3, p_std=0.1, type='distance'):
        Tsample_list = []
        qsample_list = []
        for i in range(len(self.Ttrajs)):
            if type == 'gaussian':
                Tsamples = se3.Gaussian_sampling_SE3(self.Ttrajs[i], w_std=w_std, p_std=p_std, batch_size=batch_size)
            elif type == 'distance':
                sampler = curve.DistanceGaussianSamplerSE3(self.Ttrajs[i], w_std, p_std, 10)
                Tsamples = sampler.sample(batch_size)
            wsamples = lie.log_SO3_T(Tsamples)
            qsamples = lie.screw_bracket(wsamples)
            Tsample_list.append(Tsamples.unsqueeze(0))
            qsample_list.append(qsamples.unsqueeze(0))
        self.Tsamples = torch.cat(Tsample_list, dim=0).to(self.device)
        self.qsamples = torch.cat(qsample_list, dim=0).to(self.device)
    
    def save_samples(self, file_root, file_name):
        path = os.path.join(file_root, file_name)
        samples = {'Tsamples' : np.array(self.Tsamples.detach().cpu()), 'qsamples' : np.array(self.qsamples.cpu())}
        scipy.io.savemat(path, samples)
    
    def load_samples(self, file_root, file_name):
        path = os.path.join(file_root, file_name)
        samples = scipy.io.loadmat(path)
        self.Tsamples = torch.tensor(samples['Tsamples']).to(self.device)
        self.qsamples = torch.tensor(samples['qsamples']).to(self.device)
    
    def generate_sample_trajectory(self, data_num=None, eta_R=1, eta_p=1, time_step=150):
        if data_num is None:
            data_range = range(len(self.Ttrajs))
        else:
            data_range = [data_num]
        
        Tsamples_traj_list = []
        for d_num in tqdm.tqdm(data_range, desc='Generating sample trajs'):
            with torch.no_grad():
                traj = self.SE3_traj_gen(self.Tsamples[d_num], d_num, eta_R, eta_p, dt=self.dts[d_num], time_step=time_step)
            Tsamples_traj_list.append(traj.unsqueeze(0))
        self.Tsample_trajs = torch.cat(Tsamples_traj_list, dim=0)
        
        if torch.isnan(self.Tsample_trajs).int().sum() > 0:
            print("nan in Tsample_trajs")

    def SE3_traj_gen(self, T, data_num, eta_R, eta_p, dt, time_step):
        Ttraj_list = [T.unsqueeze(1),]
        for i in range(time_step-1):
            Tdot = self.model_forward(T, data_num, eta_R, eta_p)
            T_new = se3.SE3_update_batch(T, Tdot, dt) 
            T = T_new
            Ttraj_list.append(T.unsqueeze(1))
        Ttraj = torch.cat(Ttraj_list, dim=1)
        return Ttraj
    
    def fit_traj_error(self, data_num=None, eta_R=1, eta_p=1):
        if data_num is None:
            data_range = range(len(self.Ttrajs))
        else:
            data_range = [data_num]
        error_list = []
        for d_num in data_range:
            wbpdot_out = self.model_forward(data_num=d_num, T_input=self.Ttrajs[d_num], eta_R=eta_R, eta_p=eta_p, output_type='wbpdot')
            wbpdot_traj = se3.Tdot_to_wbpdot(self.Tdottrajs[d_num], self.Ttrajs[d_num])
            error = ((wbpdot_out - wbpdot_traj)).norm(dim=-1).mean()
            wbpdot_max = wbpdot_traj.norm(dim=-1).max()
            error_list.append(error/wbpdot_max)
        
        total_error_std, total_error_mean = torch.std_mean(torch.tensor(error_list))
        return total_error_std.detach().numpy().tolist(), total_error_mean.detach().numpy().tolist()
    
    def mimic_error(self, data_num, eta_R, eta_p):
        if data_num is None:
            data_range = range(len(self.Ttrajs))
        else:
            data_range = [data_num]
            
        parallel_error_list = []
        for d_num in tqdm.tqdm(data_range, desc='Calculating parallel errors'):
            wbpdot_parallel = se3.BCSDM_SE3(self.Tsamples[d_num], 0, 0, self.Ttrajs[d_num], self.Tdottrajs[d_num], c=0, d=100, version='wbpdot')
            wbpdot_output = self.model_forward(self.Tsamples[d_num], d_num, eta_R, eta_p, output_type = 'wbpdot')
            wbpdot_traj = se3.Tdot_to_wbpdot(self.Tdottrajs[d_num], self.Ttrajs[d_num])
            wbpdot_max = wbpdot_traj.norm(dim=-1).max()
            parallel_error = ((wbpdot_parallel - wbpdot_output)**2).norm(dim=-1).mean()
            parallel_error_list.append(parallel_error / wbpdot_max)
        
        parallel_error_std, parallel_error_mean = torch.std_mean(torch.tensor(parallel_error_list))
        
        return parallel_error_std.detach().numpy().tolist(), parallel_error_mean.detach().numpy().tolist()
    
    def lyapunov_exp(self, data_num, eps):
        if data_num is None:
            data_range = range(len(self.Ttraj_list))
        else:
            data_range = [data_num]
        
        lamb_list = []
        for d_num in tqdm.tqdm(data_range, desc='Calculating lyapunov exponents'):
            dist_traj = curve.get_closest_dist_traj_SE3(self.Tsample_trajs[d_num], self.Ttrajs[d_num])
            if torch.isnan(dist_traj).int().sum() > 0:
                print("nan in dist_traj")
            lamb = curve.cal_lyapunov_exponent(dist_traj, eps=eps)
            lamb_list.append(lamb.unsqueeze(0))
        
        lamb = torch.cat(lamb_list, dim=0)
        lamb_std, lamb_mean = torch.std_mean(lamb.mean(dim=-1))
        return lamb_std.detach().numpy().tolist(), lamb_mean.detach().numpy().tolist()
    
    def cvf_mvf_error(self, data_num):
        if data_num is None:
            data_range = range(len(self.Ttrajs))
        else:
            data_range = [data_num]
        
        mvf_error_list = []
        cvf_error_list = []
        for d_num in data_range:
            self.Tsamples[d_num]
            
            Tdot_parallel = se3.BCSDM_SE3(self.Tsamples[d_num],
                                                     0, 0,
                                                     self.Ttrajs[d_num],
                                                     self.Tdottrajs[d_num],
                                                     version='Tdot')
            Tdot_contract = se3.BCSDM_SE3(self.Tsamples[d_num],
                                                     torch.inf, torch.inf,
                                                     self.Ttrajs[d_num],
                                                     self.Tdottrajs[d_num],
                                                     version='Tdot')
            Tdot_parallel_out = self.model_forward(self.Tsamples[d_num], d_num, 0, 0, output_type='Tdot')
            Tdot_contract_out = self.model_forward(self.Tsamples[d_num], d_num, torch.inf, torch.inf, output_type='Tdot')
            
            Tdot_parallel = Tdot_parallel[:,:3,:].reshape(len(self.Tsamples[d_num]), -1)
            Tdot_contract = Tdot_contract[:,:3,:].reshape(len(self.Tsamples[d_num]), -1)
            Tdot_parallel_out = Tdot_parallel_out[:,:3,:].reshape(len(self.Tsamples[d_num]), -1)
            Tdot_contract_out = Tdot_contract_out[:,:3,:].reshape(len(self.Tsamples[d_num]), -1)
            
            mvf_error = ((Tdot_parallel - Tdot_parallel_out)**2).sum(dim=1).sqrt().mean()
            cvf_error = ((Tdot_contract - Tdot_contract_out)**2).sum(dim=1).sqrt().mean()
            mvf_error_list.append(mvf_error)
            cvf_error_list.append(cvf_error)
        
        mvf_error_std, mvf_error_mean = torch.std_mean(torch.tensor(mvf_error_list))
        cvf_error_std, cvf_error_mean = torch.std_mean(torch.tensor(cvf_error_list))
        
        mvf_std = mvf_error_std.detach().numpy().tolist()
        mvf_mean = mvf_error_mean.detach().numpy().tolist()
        cvf_std = cvf_error_std.detach().numpy().tolist()
        cvf_mean = cvf_error_mean.detach().numpy().tolist()
        
        return cvf_std, cvf_mean, mvf_std, mvf_mean