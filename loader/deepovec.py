import torch
from torch.utils.data import Dataset


class deepovec(Dataset):
    def __init__(self,
                 root,
                 split='training',
                 split_ratio_train_val_test = (7/10, 2/10, 1/10),
                 **kwargs):
        super().__init__()
        
        self.split = split
        (xtraj_tuple, xdottraj_tuple, sample_tuple) = torch.load(root)
        (xsample, parallel_v, contraction_v, latent_v) = sample_tuple
        xsample = xsample.to(torch.float32)
        parallel_v = parallel_v.to(torch.float32)
        contraction_v = contraction_v.to(torch.float32)
        latent_v = latent_v.to(torch.float32)
        num_train_data = int(len(xsample) * split_ratio_train_val_test[0])
        num_valid_data = int(len(xsample) * split_ratio_train_val_test[1]) 
        
        if split == 'training':
            self.x_data = xsample[:num_train_data]
            self.parallel_v_data = parallel_v[:num_train_data]
            self.contraction_v_data = contraction_v[:num_train_data]
            self.latent_v_data = latent_v[:num_train_data]
        if split == 'validation':
            self.x_data = xsample[num_train_data:num_train_data + num_valid_data]
            self.parallel_v_data = parallel_v[num_train_data:num_train_data + num_valid_data]
            self.contraction_v_data = contraction_v[num_train_data:num_train_data + num_valid_data]
            self.latent_v_data = latent_v[num_train_data:num_train_data + num_valid_data]
        if split == 'test':
            self.x_data = xsample[num_train_data + num_valid_data:]
            self.parallel_v_data = parallel_v[num_train_data + num_valid_data:]
            self.contraction_v_data = contraction_v[num_train_data + num_valid_data:]
            self.latent_v_data = latent_v[num_train_data + num_valid_data:]
        if split == 'all':
            self.x_data = xsample
            self.parallel_v_data = parallel_v
            self.contraction_v_data = contraction_v
            self.latent_v_data = latent_v
        if split == 'visualization':
            self.x_data = xsample[num_train_data + num_valid_data:]
            self.parallel_v_data = parallel_v[num_train_data + num_valid_data:]
            self.contraction_v_data = contraction_v[num_train_data + num_valid_data:]
        
        self.xtraj_tuple = []
        self.xdottraj_tuple = []
        for xtraj in xtraj_tuple:
            self.xtraj_tuple.append(xtraj.to(torch.float32))
        for xdottraj in xdottraj_tuple:
            self.xdottraj_tuple.append(xdottraj.to(torch.float32))
        self.xtraj_tuple = tuple(self.xtraj_tuple)
        self.xdottraj_tuple = tuple(self.xdottraj_tuple)
        

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx].detach()
        
        if self.split == 'visualization':
            xtraj = self.xtraj_tuple
            return x, xtraj
        else:
            parallel_v = self.parallel_v_data[idx].detach()
            contraction_v = self.contraction_v_data[idx].detach()
            xtraj = self.xtraj_tuple[int(self.latent_v_data[idx])].detach()
            xdottraj = self.xdottraj_tuple[int(self.latent_v_data[idx])].detach()
            return x, parallel_v, contraction_v, xtraj, xdottraj