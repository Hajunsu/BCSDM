import torch
import torch.nn as nn

class Base_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cpu')

    def forward(self, *args, **kwargs):
        raise(NotImplementedError)
    
    def train_step(self,*args, **kwargs):
        raise(NotImplementedError)
    
    def visualization_step(self, *args, **kwargs):
        raise(NotImplementedError)
    
    def to(self, *args, **kwargs):
        if type(args[0]) == torch.device:
            self.device == args[0]
        elif hasattr(args[0], 'device'):
            self.device = args[0].device
        elif type(args[0]) == str:
            if 'cuda' in args[0]:
                if '0' in args[0]:
                    self.device = torch.device('cuda:0')
                elif '1' in args[0]:
                    self.device = torch.device('cuda:1')
                else:
                    self.device = torch.device('cuda')
            elif 'cpu' in args[0]:
                self.device = torch.device('cpu')
        return super().to(*args, **kwargs)