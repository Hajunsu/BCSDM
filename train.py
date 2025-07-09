import os
import torch
import wandb
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from utils.utils import save_yaml
from models import get_model
from trainers import get_trainer, get_logger
from loader import get_dataloader
from optimizers import get_optimizer
from utils.utils import parse_unknown_args, parse_nested_args


def run(cfg, writer, copied_yml_path):
    # Setup seeds
    seed = cfg.training.get('seed', 0)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup device
    device = cfg.device
    
    # Setup Dataloader
    d_dataloaders = {}
    for key, dataloader_cfg in cfg.data.items():
        d_dataloaders[key] = get_dataloader(dataloader_cfg)
    if 'traj' in d_dataloaders.keys():
        xstable = d_dataloaders['traj'].dataset.xstable
        cfg['xstable'] = xstable
    
    model = get_model(cfg).to(device)
    logger = get_logger(cfg, writer)

    print('model :', model)
    print('Parameters :', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Setup optimizer
    optimizer = get_optimizer(cfg.training.optimizer, model.parameters())
    scheduler_mode = cfg.training.get("scheduler", None)
    if scheduler_mode == "cosine":
        print("cosine scheduler mode")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.n_epoch)
    elif scheduler_mode == "multi_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.training.milestones, gamma=cfg.training.gamma)
    else:
        scheduler = None
    
    # Setup Trainer
    trainer = get_trainer(optimizer, cfg, scheduler=scheduler)
    
    # Save config file
    save_yaml(copied_yml_path, OmegaConf.to_yaml(cfg))
    print(f"config saved as {copied_yml_path}")
    
    # Train
    model, train_result = trainer.train(
        model,
        d_dataloaders,
        logger=logger,
        logdir=writer.file_writer.get_logdir(),
    )
    
    print("end code")

    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./configs/SE3/se3.yml') # type=str)
    parser.add_argument("--device", default='0')
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--run", default=None)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)
    
    print(args.config)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    # set device
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    if args.device == "cpu":
        cfg["device"] = f"cpu"
    else:
        cfg["device"] = "cuda:0"

    if args.run is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        run_id = args.run

    config_basename = os.path.basename(args.config).split(".")[0]
    []
    if hasattr(cfg, "logdir"):
        logdir = cfg["logdir"]
    else:
        logdir = args.logdir
    logdir = os.path.join(logdir, run_id)
    if os.path.exists(logdir):
        logdir = logdir + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = str(Path(logdir))
    
    writer = SummaryWriter(logdir=logdir)  # tensorboard
    print("Result directory: {}".format(logdir))

    # copy config file
    copied_yml_path = os.path.join(logdir, os.path.basename(args.config))
    copied_yml_path = Path(copied_yml_path)
    
    if 'entity' in cfg.keys():
        wandb.init(
            entity=cfg['entity'],
            project=cfg['wandb_project_name'],
            config=OmegaConf.to_container(cfg),
            name=logdir
        )
    
    cfg = run(cfg, writer, copied_yml_path)