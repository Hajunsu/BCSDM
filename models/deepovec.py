import torch
from re import X

from .base_model import Base_model
from utils import S2_functions as s2
from utils import SE3_functions as se3


class DeepOVec_euc(Base_model):
    def __init__(self, deeponet_parallel, deeponet_contract, gamma, *args, **kwargs):
        super().__init__()
        self.deeponet_parallel = deeponet_parallel
        self.deeponet_contract = deeponet_contract
        self.gamma = gamma
    
    def get_xdots(self, x, xtraj):
        xtraj_flat = xtraj.reshape(len(x),-1)
        xdot_parallel = self.deeponet_parallel(xtraj_flat.to(torch.float32), x.to(torch.float32))
        xdot_contract = self.deeponet_contract(xtraj_flat.to(torch.float32), x.to(torch.float32))
        return xdot_parallel, xdot_contract
    
    def forward(self, x, xtraj, eta):
        xdot_parallel, xdot_contract = self.get_xdots(x, xtraj)
        if eta == torch.inf:
            xdot = xdot_contract
        else:
            xdot = xdot_parallel + eta * xdot_contract  
        return xdot
    
    def get_loss(self, x, xtraj, xdottraj, xdot_parallel, xdot_contract):
        xdot_parallel_out, xdot_contract_out = self.get_xdots(x, xtraj)
        loss_parallel = ((xdot_parallel - xdot_parallel_out) ** 2).view(len(xdot_parallel), -1).mean(dim=1).mean()
        loss_contract = ((xdot_contract - xdot_contract_out) ** 2).view(len(xdot_contract), -1).mean(dim=1).mean()
        
        xdot_parallel_traj, xdot_contract_traj = self.get_xdots(xtraj[0], xtraj[0].unsqueeze(0).repeat(len(xtraj[0]),1,1))
        loss_traj_parallel = ((xdot_parallel_traj - xdottraj) ** 2).view(len(xdot_parallel_traj), -1).mean(dim=1).mean()
        loss_traj_contract = (xdot_contract_traj**2).view(len(xdot_contract_traj), -1).mean(dim=1).mean()
        loss_fitting = loss_traj_parallel + loss_traj_contract
        
        loss = loss_parallel + loss_contract + self.gamma * loss_fitting
        
        loss_dict = {}
        loss_dict["loss"] = loss
        loss_dict["loss_parallel"] = loss_parallel
        loss_dict["loss_contract"] = loss_contract
        loss_dict["loss_fitting"] = loss_fitting
        
        return loss_dict
    
    def train_step(self, x, xtraj, xdottraj, xdot_parallel, xdot_contract, optimizer, **kwargs):
        optimizer.zero_grad()
        loss_dict = self.get_loss(x, xtraj, xdottraj, xdot_parallel, xdot_contract)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(),
                "train_loss_parallel_": loss_dict['loss_parallel'].item(),
                "train_loss_contract_": loss_dict['loss_contract'].item(),
                "train_loss_fitting_": loss_dict['loss_fitting'].item(),
                }

    def validation_step(self, x, xtraj, xdottraj, xdot_parallel, xdot_contract,**kwargs):
        loss_dict = self.get_loss(x, xtraj, xdottraj, xdot_parallel, xdot_contract)
        return {"loss": loss_dict['loss'].item(),
                "val_loss_parallel_": loss_dict['loss_parallel'].item(),
                "val_loss_contract_": loss_dict['loss_contract'].item(),
                "val_loss_fitting_": loss_dict['loss_fitting'].item(),
                }
    

class DeepOVec_S2(Base_model):
    def __init__(self, deeponet_parallel, deeponet_contract, gamma, *args, **kwargs):
        super().__init__()
        self.deeponet_parallel = deeponet_parallel
        self.deeponet_contract = deeponet_contract
        self.gamma = gamma
    
    def get_xdots(self, x, xtraj):
        xtraj = xtraj.to(torch.float32)
        xtraj_flat = xtraj.reshape(len(x),-1)
        x = x.to(torch.float32)
        xdot_parallel = self.deeponet_parallel(xtraj_flat, x)
        xdot_contract = self.deeponet_contract(xtraj_flat, x)
        xdot_parallel_proj = s2.xdot_projection(xdot_parallel, x)
        xdot_contract_proj = s2.xdot_projection(xdot_contract, x)
        return xdot_parallel_proj, xdot_contract_proj
    
    def forward(self, x, xtraj, eta):
        xdot_parallel, xdot_contract = self.get_xdots(x, xtraj)
        if eta == torch.inf:
            xdot = xdot_contract
        else:
            xdot = xdot_parallel + eta * xdot_contract  
        return xdot
    
    def get_loss(self, x, xtraj, xdottraj, xdot_parallel, xdot_contract):
        xdot_parallel_out, xdot_contract_out = self.get_xdots(x, xtraj)
        xdot_parallel_traj_out, xdot_contract_traj_out = self.get_xdots(
                                xtraj[0], xtraj[0].unsqueeze(0).repeat(len(xtraj[0]),1,1))
        
        loss_parallel = ((xdot_parallel - xdot_parallel_out) ** 2).view(len(xdot_parallel), -1).mean(dim=1).mean()
        loss_contract = ((xdot_contract - xdot_contract_out) ** 2).view(len(xdot_contract), -1).mean(dim=1).mean()
        loss_traj_parallel = ((xdot_parallel_traj_out - xdottraj[0]) ** 2).mean(dim=1).mean()
        loss_traj_contract = ((xdot_contract_traj_out) ** 2).mean(dim=1).mean()
        
        loss = (loss_parallel + loss_contract) + self.gamma * (loss_traj_parallel + loss_traj_contract)
        
        loss_dict = {}
        loss_dict["loss"] = loss
        loss_dict["loss_parallel"] = loss_parallel
        loss_dict["loss_contract"] = loss_contract
        loss_dict["loss_traj_fitting_parallel"] = loss_traj_parallel
        loss_dict["loss_traj_fitting_contract"] = loss_traj_contract
        
        return loss_dict
    
    def train_step(self, x, xtraj, xdottraj, xdot_parallel, xdot_contract, optimizer, **kwargs):
        optimizer.zero_grad()
        loss_dict = self.get_loss(x, xtraj, xdottraj, xdot_parallel, xdot_contract)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(),
                "train_loss_parallel_": loss_dict['loss_parallel'].item(),
                "train_loss_contract_": loss_dict['loss_contract'].item(),
                "loss_traj_fitting_parallel_": loss_dict['loss_traj_fitting_parallel'].item(),
                "loss_traj_fitting_contract_": loss_dict['loss_traj_fitting_contract'].item(),
                }

    def validation_step(self, x, xtraj, xdottraj, xdot_parallel, xdot_contract,**kwargs):
        loss_dict = self.get_loss(x, xtraj, xdottraj, xdot_parallel, xdot_contract)
        return {"loss": loss_dict['loss'].item(),
                "val_loss_parallel_": loss_dict['loss_parallel'].item(),
                "val_loss_contract_": loss_dict['loss_contract'].item(),
                "loss_traj_fitting_parallel_": loss_dict['loss_traj_fitting_parallel'].item(),
                "loss_traj_fitting_contract_": loss_dict['loss_traj_fitting_contract'].item(),
                }


class DeepOVec_SE3(Base_model):
    def __init__(self, deeponet_parallel, deeponet_contract, gamma=1, *args, **kwargs):
        super().__init__()
        self.deeponet_parallel = deeponet_parallel
        self.deeponet_contract = deeponet_contract
        self.gamma = gamma

    def get_Tdots(self, T, Ttraj):
        batch = len(T)
        T_flat = se3.mat2vec_SE3(T).to(T)
        Ttraj_flat = se3.mat2vec_SE3_traj(Ttraj).to(T)
        Tdot_parallel = self.deeponet_parallel(Ttraj_flat, T_flat)
        Tdot_contract = self.deeponet_contract(Ttraj_flat, T_flat)
        Tdot_parallel = torch.cat([Tdot_parallel[:,:9].reshape(batch,3,3),Tdot_parallel[:,9:].unsqueeze(-1)], dim=2)
        Tdot_contract = torch.cat([Tdot_contract[:,:9].reshape(batch,3,3),Tdot_contract[:,9:].unsqueeze(-1)], dim=2)
        Tdot_parallel[:,:3,:3] = se3.Rdot_projection(Tdot_parallel[:,:3,:3], T[:,:3,:3])
        Tdot_contract[:,:3,:3] = se3.Rdot_projection(Tdot_contract[:,:3,:3], T[:,:3,:3])
        return Tdot_parallel, Tdot_contract
    
    def forward(self, T, Ttraj, eta_R, eta_p):
        batch = len(T)
        Tdot_parallel, Tdot_contract = self.get_Tdots(T, Ttraj)
        Tdot = torch.zeros(4,4).unsqueeze(0).repeat(batch,1,1).to(T)
        
        if eta_R == torch.inf:
            Tdot[:,:3,:3] = Tdot_contract[:,:3,:3]
        else:
            Tdot[:,:3,:3] = Tdot_parallel[:,:3,:3] + eta_R * Tdot_contract[:,:3,:3]
        
        if eta_p == torch.inf:
            Tdot[:,:3,3] = Tdot_contract[:,:3,3]
        else:
            Tdot[:,:3,3] = Tdot_parallel[:,:3,3] + eta_p * Tdot_contract[:,:3,3]
        return Tdot
    
    def get_loss(self, T, Ttraj, Tdottraj, Tdot_parallel, Tdot_contract):
        Tdot_parallel = se3.mat2vec_SE3(Tdot_parallel).to(T)
        Tdot_contract = se3.mat2vec_SE3(Tdot_contract).to(T)
        Tdot_parallel_out, Tdot_contract_out = self.get_Tdots(T, Ttraj)
        Tdot_parallel_out = se3.mat2vec_SE3(Tdot_parallel_out)
        Tdot_contract_out = se3.mat2vec_SE3(Tdot_contract_out)
        
        Tdot_parallel_traj_out, Tdot_contract_traj_out = self.get_Tdots(Ttraj[0], Ttraj[0].unsqueeze(0).repeat(len(Ttraj[0]),1,1,1))
        Tdot_parallel_traj_out = se3.mat2vec_SE3(Tdot_parallel_traj_out)
        Tdot_contract_traj_out = se3.mat2vec_SE3(Tdot_contract_traj_out)
        
        loss_parallel = ((Tdot_parallel - Tdot_parallel_out)**2).mean()
        loss_contract = ((Tdot_contract - Tdot_contract_out)**2).mean()
        
        # MSE traj
        Tdottraj_flat = se3.mat2vec_SE3(Tdottraj[0]).to(T)
        loss_parallel_traj = ((Tdot_parallel_traj_out - Tdottraj_flat)**2).mean()
        loss_contract_traj = ((Tdot_contract_traj_out)**2).mean()
        
        loss_fitting = loss_parallel_traj + loss_contract_traj
        loss = (loss_parallel + loss_contract) + self.gamma * loss_fitting
        
        loss_dict = {}
        loss_dict["loss"] = loss
        loss_dict["loss_parallel"] = loss_parallel
        loss_dict["loss_contract"] = loss_contract
        loss_dict["loss_parallel_traj"] = loss_parallel_traj
        loss_dict["loss_contract_traj"] = loss_contract_traj
        
        return loss_dict
    
    def train_step(self, T, Ttraj, Tdottraj, Tdot_parallel, Tdot_contract, optimizer, **kwargs):
        optimizer.zero_grad()
        loss_dict = self.get_loss(T, Ttraj, Tdottraj, Tdot_parallel, Tdot_contract)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(),
                "train_loss_parallel_": loss_dict['loss_parallel'].item(),
                "train_loss_contract_": loss_dict['loss_contract'].item(),
                "train_loss_parallel_traj_": loss_dict['loss_parallel_traj'].item(),
                "train_loss_contract_traj_": loss_dict['loss_contract_traj'].item(),
                }

    def validation_step(self, T, Ttraj, Tdottraj, Tdot_parallel, Tdot_contract,**kwargs):
        loss_dict = self.get_loss(T, Ttraj, Tdottraj, Tdot_parallel, Tdot_contract)
        return {"loss": loss_dict['loss'].item(),
                "val_loss_parallel_": loss_dict['loss_parallel'].item(),
                "val_loss_contract_": loss_dict['loss_contract'].item(),
                "val_loss_parallel_traj_": loss_dict['loss_parallel_traj'].item(),
                "val_loss_contract_traj_": loss_dict['loss_contract_traj'].item(),
                }