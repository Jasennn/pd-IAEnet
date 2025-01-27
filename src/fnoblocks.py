import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


KERNEL_SIZE = 1
class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, activation=F.relu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activation = activation
        
        self.conv = SpectralConv2d(in_channels, out_channels, modes, modes)
        self.w = nn.Conv1d(in_channels, out_channels, KERNEL_SIZE)
    
    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        spatial_shape = [x.shape[2], x.shape[3]]
        
        x1 = self.conv(x)
        
        x2 = self.w(x.reshape(batch_size, self.in_channels, -1))
        x2 = x2.reshape(batch_size, self.out_channels, spatial_shape[0], spatial_shape[1])
        
        x = x1 + x2
        del x1, x2
        
        if self.activation is not None:
            x = self.activation(x)
            
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
         
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
    
    def compl_mul2d(self, input, weights):
        return torch.einsum('bixy,ioxy->boxy', input, weights)
    
    def forward(self, x: Tensor):
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply low frequency components with weights.
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x