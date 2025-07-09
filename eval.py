import os
import torch
import argparse
from pathlib import Path
from datetime import datetime
from pprint import pprint as pp
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from utils.utils import save_yaml
from train import parse_unknown_args, parse_nested_args
from utils.S2_metric import S2_metric
from utils.SE3_metric import SE3_metric


def S2_eval(cfg):
    results = {}
    results['data num'] = 'all' if cfg.data_num is None else cfg.data_num
    
    # load model ###
    Model = S2_metric(root=Path(cfg.model.root),
                    identifier=cfg.model.identifier,
                    config_file=cfg.model.config_file,
                    ckpt_file=cfg.model.ckpt_file,
                    model_type=cfg.model.type,
                    demo_length=cfg.model.demo_length,
                    device=cfg.device)
    print('model name :', cfg.model.identifier)
    if cfg.model.params:
        print('params :', Model.params)
        results['params'] = Model.params
    
    ### sampling ###
    if cfg.sampling.sample:
        Model.sampling_traj(cfg.sampling.batch, cfg.sampling.std, cfg.sampling.type)
        Model.save_samples(Path(cfg.sampling.file_root), cfg.sampling.file_name)
    Model.load_samples(Path(cfg.sampling.file_root), cfg.sampling.file_name)

    ### fiiting error ###
    if cfg.fit_traj:
        fit_error_traj_std, fit_error_traj_mean = Model.fit_traj_error(cfg.data_num, cfg.eta)
        print(f'fitting error traj mean : {fit_error_traj_mean:.6f}')
        print(f'fitting error traj std : {fit_error_traj_std:.6f}')
        results['fitting error traj mean'] = fit_error_traj_mean
        results['fitting error traj std'] = fit_error_traj_std
    if cfg.fit_all:
        fit_error_all_std, fit_error_all_mean = Model.fit_all_error(cfg.data_num, cfg.eta)
        print(f'fitting error all mean : {fit_error_all_mean:.6f}')
        print(f'fitting error all std : {fit_error_all_std:.6f}')
        results['fitting error all mean'] = fit_error_all_mean
        results['fitting error all std'] = fit_error_all_std
        pass
    
    ### generate sample trajectory ###
    if cfg.contraction:
        Model.generate_sample_trajectory(cfg.data_num, cfg.eta, cfg.traj_length)
    
    ### paralllel ###
    if cfg.mimic:
        mimic_error_std, mimic_error_mean = Model.mimic_error(cfg.data_num, cfg.eta, cfg.mimicking)
        print('mimic error mean :', mimic_error_mean)
        print('mimic error std :', mimic_error_std)
        results['mimic error mean'] = mimic_error_mean
        results['mimic error std'] = mimic_error_std
    
    ### contraction ###
    if cfg.contraction:
        lamb_std, lamb_mean = Model.lyapunov_exp(cfg.data_num, cfg.eps)
        print('lamb mean :', lamb_mean)
        print('lamb std :', lamb_std)
        results['lamb mean'] = lamb_mean
        results['lamb std'] = lamb_std
    
    ### mvf, cvf error ###
    if cfg.cvf_mvf:
        cvf_std, cvf_mean, mvf_std, mvf_mean = Model.cvf_mvf_error(cfg.data_num)
        print('cvf mean :', cvf_mean)
        print('cvf std :', cvf_std)
        results['cvf mean'] = cvf_mean
        results['cvf std'] = cvf_std
        print('mvf mean :', mvf_mean)
        print('mvf std :', mvf_std)
        results['mvf mean'] = mvf_mean
        results['mvf std'] = mvf_std
        
    ### save results ###
    results_path_torch = os.path.join(cfg.logdir, 'results.pt')
    results_path_yml = os.path.join(cfg.logdir, 'results.yml')
    torch.save(results, results_path_torch)
    save_yaml(results_path_yml, OmegaConf.to_yaml(results))
    print(f"results saved as {results_path_yml}")
    
    ### visualization ###
    if cfg.vis_local or cfg.vis_sphere:
        file_name = os.path.join(cfg.logdir, cfg.model.type)
        Model.vis(cfg.vis.vis_data_num, cfg.eta, cfg.vis_local, cfg.vis_sphere, file_name)

def SE3_eval(cfg):
    results = {}
    results['data num'] = 'all' if cfg.data_num is None else cfg.data_num
    
    # load model ###
    Model = SE3_metric(Path(cfg.model.root),
                     cfg.model.identifier,
                     cfg.model.config_file,
                     cfg.model.ckpt_file,
                     model_type=cfg.model.type,
                     device=cfg.device)
    print('model name :', cfg.model.identifier)
    if cfg.model.params:
        print('params :', Model.params)
        results['params'] = Model.params
    
    ### sampling ###
    if cfg.sampling.sample:
        Model.sampling_traj(cfg.sampling.batch, cfg.sampling.w_std, cfg.sampling.p_std, cfg.sampling.type)
        Model.save_samples(Path(cfg.sampling.file_root), cfg.sampling.file_name)
    Model.load_samples(Path(cfg.sampling.file_root), cfg.sampling.file_name)

    ### fiiting error ###
    if cfg.fit_traj:
        fit_error_traj_std, fit_error_traj_mean = Model.fit_traj_error(cfg.data_num, cfg.eta_R, cfg.eta_p)
        print(f'fitting error traj mean : {fit_error_traj_mean:.6f}')
        print(f'fitting error traj std : {fit_error_traj_std:.6f}')
        results['fitting error traj mean'] = fit_error_traj_mean
        results['fitting error traj std'] = fit_error_traj_std
    
    ### generate sample trajectory ###
    if cfg.contraction:
        Model.generate_sample_trajectory(cfg.data_num, cfg.eta_R, cfg.eta_p, cfg.time_step)
    
    ### paralllel ###
    if cfg.mimic:
        mimic_error_std, mimic_error_mean = Model.mimic_error(cfg.data_num, cfg.eta_R, cfg.eta_p)
        print('mimic error mean :', mimic_error_mean)
        print('mimic error std :', mimic_error_std)
        results['mimic error mean'] = mimic_error_mean
        results['mimic error std'] = mimic_error_std

    ### contraction ###
    if cfg.contraction:
        lamb_std, lamb_mean = Model.lyapunov_exp(cfg.data_num, cfg.eps)
        print('lamb mean :', lamb_mean)
        print('lamb std :', lamb_std)
        results['lamb mean'] = lamb_mean
        results['lamb std'] = lamb_std
    
    ### mvf, cvf error ###
    if cfg.cvf_mvf:
        cvf_std, cvf_mean, mvf_std, mvf_mean = Model.cvf_mvf_error(cfg.data_num)
        print('cvf mean :', cvf_mean)
        print('cvf std :', cvf_std)
        results['cvf mean'] = cvf_mean
        results['cvf std'] = cvf_std
        print('mvf mean :', mvf_mean)
        print('mvf std :', mvf_std)
        results['mvf mean'] = mvf_mean
        results['mvf std'] = mvf_std

    ### save results ###
    results_path_torch = os.path.join(cfg.logdir, 'results.pt')
    results_path_yml = os.path.join(cfg.logdir, 'results.yml')
    torch.save(results, results_path_torch)
    save_yaml(results_path_yml, OmegaConf.to_yaml(results))
    print(f"results saved as {results_path_yml}")


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./configs/eval/S2_bc-deepovec.yml', type=str)
    parser.add_argument("--device", default='0')
    parser.add_argument("--fit_traj", action="store_true")
    parser.add_argument("--fit_all", action="store_true")
    parser.add_argument("--mimic", action="store_true")
    parser.add_argument("--contraction", action="store_true")
    parser.add_argument("--cvf_mvf", action="store_true")
    parser.add_argument("--vis_local", action="store_true")
    parser.add_argument("--vis_sphere", action="store_true")
    parser.add_argument("--run", default=None)
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, vars(args))
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    pp(d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))
    
    # set device
    if args.device == "cpu":
        cfg["device"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
        cfg["device"] = "cuda:0"

    # run_id
    if args.run is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        run_id = args.run
    
    # log_dir
    config_basename = os.path.basename(args.config).split(".")[0]
    # []
    if hasattr(cfg, "logdir"):
        logdir = cfg["logdir"]
    else:
        logdir = args.logdir
    logdir = os.path.join(logdir, run_id)
    if os.path.exists(logdir):
        logdir = logdir + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir=logdir)  # tensorboard
    print("Result directory: {}".format(logdir))
    
    # config
    copied_yml_path = os.path.join(logdir, os.path.basename(args.config))
    save_yaml(copied_yml_path, OmegaConf.to_yaml(cfg))
    print(f"config saved as {copied_yml_path}")
    cfg = OmegaConf.merge(cfg, {'logdir': logdir})
    
    # evaluate
    if cfg.data_type == 'S2':
        S2_eval(cfg)
    elif cfg.data_type == 'SE3':
        SE3_eval(cfg)