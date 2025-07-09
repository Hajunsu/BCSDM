import os
from omegaconf import OmegaConf
import torch

from models.deepovec import (
    DeepOVec_euc,
    DeepOVec_S2,
    DeepOVec_SE3,
)

from models.modules import (
    FC_vec,
    DeepONet,
)

def get_net(in_dim, out_dim, **kwargs):
    if kwargs["arch"] == "fc_vec":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_vec(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] in ["deepovec_euc",
                            "deepovec_S2",
                            "deepovec_SE3",]:
        branch_arch = kwargs["branch"]
        trunk_arch = kwargs["trunk"]
        net = DeepONet(
            out_dim=out_dim,
            branch_arch=branch_arch,
            trunk_arch=trunk_arch,
        )
        
    return net

def get_vf(data, **model_cfg):
    in_dim = model_cfg.get('in_dim')
    total_out_dim = model_cfg.get('total_out_dim')
    arch = model_cfg.get('arch')
    gamma = model_cfg.get('gamma', 1)
    
    if arch == 'deepovec_euc':
        deeponet_parallel = get_net(in_dim=in_dim, out_dim=total_out_dim, **model_cfg)
        deeponet_contract = get_net(in_dim=in_dim, out_dim=total_out_dim, **model_cfg)
        model = DeepOVec_euc(deeponet_parallel, deeponet_contract, gamma=gamma)
    elif arch == 'deepovec_S2':
        deeponet_parallel = get_net(in_dim=in_dim, out_dim=total_out_dim, **model_cfg)
        deeponet_contract = get_net(in_dim=in_dim, out_dim=total_out_dim, **model_cfg)
        model = DeepOVec_S2(deeponet_parallel, deeponet_contract, gamma=gamma)
    elif arch == 'deepovec_SE3':
        deeponet_parallel = get_net(in_dim=in_dim, out_dim=total_out_dim, **model_cfg)
        deeponet_contract = get_net(in_dim=in_dim, out_dim=total_out_dim, **model_cfg)
        model = DeepOVec_SE3(deeponet_parallel, deeponet_contract, gamma=gamma)
    
    return model


def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
        training_dict = cfg['training']
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    data = cfg["data"]["training"]["dataset"]
    if "lyapunov_optimizer" in training_dict.keys():
        model_dict['lyapunov_optimizer'] = training_dict['lyapunov_optimizer'] 
    if "xstable" in cfg:
        model_dict['xstable'] = cfg['xstable']
    model_instance = _get_model_instance(name)
    model = model_instance(data, **model_dict)
    return model


def _get_model_instance(name):
    try:
        return {
            "deepovec_euc": get_vf,
            "deepovec_S2": get_vf,
            "deepovec_SE3": get_vf,
        }[name]
    except:
        raise ("Model {} not available".format(name))


def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    try:
        cfg = OmegaConf.load(config_path)
    except:
        print('Config path does not exist. Trying one directory below..')
        config_path = os.path.join('..', config_path)
        print(f'config_path = {config_path}')
        cfg = OmegaConf.load(config_path)
    
    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    
    return model, cfg