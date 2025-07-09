import os
import time
import tqdm
import torch
import numpy as np
from utils.utils import averageMeter


class BaseTrainer:
    def __init__(self, optimizer, training_cfg, device, scheduler=None):
        self.training_cfg = training_cfg
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, model, d_dataloaders, logger=None, logdir=""):
        cfg = self.training_cfg
    
        time_meter = averageMeter()
        train_loader, val_loader = (d_dataloaders["training"], d_dataloaders["validation"])
        traj_loader = d_dataloaders["traj"]
        traj_original_loader = d_dataloaders["traj_original"]
        kwargs = {'dataset_size': len(train_loader.dataset)}
        i_iter = 0
        best_val_loss = np.inf
        
        for i_epoch in range(1, cfg['n_epoch'] + 1):
            for ((x, xdot_parallel, xdot_contraction), (xtraj, xdottraj)) in zip(train_loader, traj_loader):
                i_iter += 1
                model.train()
                start_ts = time.time()
                d_train = model.train_step(
                    x.to(self.device), xdot_parallel.to(self.device), 
                    xdot_contraction.to(self.device), 
                    xtraj.to(self.device), xdottraj.to(self.device),
                    optimizer=self.optimizer, **kwargs)
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i_iter % cfg.print_interval == 0:
                    d_train = logger.summary_train(i_iter)
                    print(
                        f"Epoch [{i_epoch:d}] \nIter [{i_iter:d}]\tAvg Loss: {d_train['loss/train_loss_']:.6f}\tElapsed time: {time_meter.sum:.4f}"
                    )
                    time_meter.reset()

                model.eval()
                if i_iter % cfg.val_interval == 0:
                    for((x, xdot_parallel, xdot_contraction), (xtraj, xdottraj)) in zip(val_loader, traj_loader):
                        d_val = model.validation_step(
                            x.to(self.device), xdot_parallel.to(self.device), 
                            xdot_contraction.to(self.device), 
                            xtraj.to(self.device), xdottraj.to(self.device),)
                        logger.process_iter_val(d_val)
                    d_val = logger.summary_val(i_iter)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if best_model:
                        print(f'Iter [{i_iter:d}] best model saved {val_loss:.6f} <= {best_val_loss:.6f}')
                        best_val_loss = val_loss
                        self.save_model(model, logdir, best=True)
                    
                if cfg.visualization.type == 'none':
                    pass
                else:
                    pass
        
        self.save_model(model, logdir, i_iter="last")
        return model, best_val_loss

    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None):
        if best:
            if i_epoch is not None:
                pkl_name = f"model_best_iepoch_{i_epoch}.pkl"
            else:
                pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{i_epoch}.pkl"
        state = {"epoch": i_epoch, "iter": i_iter, "model_state": model.state_dict()}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
    
    
class DeeponevecTrainer(BaseTrainer):
    def train(self, model, d_dataloaders, logger=None, logdir=""):
        cfg = self.training_cfg
    
        time_meter = averageMeter()
        train_loader, val_loader, vis_loader = (d_dataloaders["training"], d_dataloaders["validation"], d_dataloaders["visualization"])
        kwargs = {'dataset_size': len(train_loader.dataset)}
        i_iter = 0
        best_val_loss = np.inf
        
        bar = tqdm.tqdm(range(1, cfg['n_epoch'] + 1))
        for i_epoch in bar:
            for (x, xdot_parallel, xdot_contraction, xtraj, xdottraj) in train_loader:
                i_iter += 1
                model.train()
                start_ts = time.time()
                d_train = model.train_step(
                    x.to(self.device),
                    xtraj.to(self.device),
                    xdottraj.to(self.device),
                    xdot_parallel.to(self.device), 
                    xdot_contraction.to(self.device),
                    optimizer=self.optimizer, **kwargs)
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i_iter % cfg.print_interval == 0:
                    d_train = logger.summary_train(i_iter)
                    time_meter.reset()

                model.eval()
                if i_iter % cfg.val_interval == 0:
                    for (x, xdot_parallel, xdot_contraction, xtraj, xdottraj) in val_loader:
                        
                        d_val = model.validation_step(
                            x.to(self.device), 
                            xtraj.to(self.device),
                            xdottraj.to(self.device),
                            xdot_parallel.to(self.device), 
                            xdot_contraction.to(self.device),)
                        logger.process_iter_val(d_val)
                        break
                    d_val = logger.summary_val(i_iter)
                    val_loss = d_val['loss/val_loss_']
                    best_model = val_loss < best_val_loss

                    if best_model:
                        best_val_loss = val_loss
                        self.save_model(model, logdir, best=True)
                        best_model_last = model
                
                if i_iter % cfg.print_interval == 0 and i_iter % cfg.val_interval == 0:
                    if best_model:
                        bar.set_postfix_str(f"Loss: {d_train['loss/train_loss_']:.6f}, d_val: {d_val['loss/val_loss_']:.6f}, best model saved {val_loss:.6f} <= {best_val_loss:.6f}")
                    else:
                        bar.set_postfix_str(f"Loss: {d_train['loss/train_loss_']:.6f}, d_val: {d_val['loss/val_loss_']:.6f}")
                elif i_iter % cfg.print_interval == 0:
                    bar.set_postfix_str(f"Loss: {d_train['loss/train_loss_']:.6f}")
                
                if cfg.visualization.type == 'none':
                    pass
                else:
                    pass
            if self.scheduler is not None:
                self.scheduler.step()
            if i_epoch % cfg.get('save_epochs', 100000000) == 0:
                self.save_model(model, logdir, i_epoch=i_epoch)
                self.save_model(best_model_last, logdir, i_epoch=i_epoch, best=True)
        
        self.save_model(model, logdir, i_iter=f"last")
        return model, best_val_loss