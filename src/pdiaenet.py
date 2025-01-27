import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from fnoblocks import FNOBlock

SKIP_CONNECTION_KERNEL_SIZE = 1
DIM = 2

FNO_POSTPROCESSING = True
if FNO_POSTPROCESSING:
    FNO_MODES = 12
REDUCER_HIDDEN_WIDTH = 128
class PDIAE_Net(nn.Module):
    def __init__(self, in_channels, out_channels, width=64, k_rank=2, modes=12, num_blocks=4):
        super().__init__()
        
        self.channel_raiser = nn.Linear(in_channels, width)
        
        self.blocks = nn.ModuleList()
        for block_idx in range(num_blocks - 1):
            self.blocks.append(PDIAE_MultiChannel(width, width, k_rank, modes, block_idx + 1, activation=F.relu))
        self.blocks.append(PDIAE_MultiChannel(width, width, k_rank, modes, num_blocks, activation=None))
        
        self.skip_connection_layer = nn.Conv1d(width * (num_blocks + 1), width, SKIP_CONNECTION_KERNEL_SIZE)
        
        self.fno_postprocess = nn.Sequential(
            FNOBlock(width, width, FNO_MODES, activation=F.relu),
            FNOBlock(width, width, FNO_MODES, activation=None)
        )
        
        self.channel_reducer1 = nn.Linear(width, REDUCER_HIDDEN_WIDTH)
        self.channel_reducer2 = nn.Linear(REDUCER_HIDDEN_WIDTH, out_channels)
        
    def forward(self, x: Tensor):
        """

        Parameters
        ----------
        x : Tensor of shape (batch_size, in_channels, s1, s2)
        """
        
        # Raise channel dimension
        x = x.movedim(1, -1) # (b, s1, s2, in_channels)
        x = self.channel_raiser(x) # (b, s1, s2, width)
        x = x.movedim(-1, 1) # (b, width, s1, s2)
        
        # Run input through sequential Multi-Channel pdIAE blocks.
        x_list = [x]
        for block in self.blocks:
            x_list = block(x_list)
            
        # Last skip connection layer.
        x = torch.cat(x_list, dim=1) # (b, (num blocks)*width, s1, s2)
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], -1) # (b, (num blocks)*width, s1*s2)
        x = self.skip_connection_layer(x) # (b, width, s1*s2)
        x = x.reshape(x_shape[0], -1, x_shape[2], x_shape[3]) # (b, width, s1, s2)
        
        # FNO Postprocessing
        if FNO_POSTPROCESSING:
            x = self.fno_postprocess(x) # (b, width, s1, s2)
        
        # Reduce channel dimension
        x = x.movedim(1, -1)
        x = self.channel_reducer1(x) # (b, s1, s2, 128)
        x = F.relu(x)
        x = self.channel_reducer2(x) # (b, s1, s2, out_channels)
        x = x.movedim(-1, 1)
        
        return x
    

IDENTITY_MAP = True
DROPOUT_PROB = 0.1
class PDIAE_MultiChannel(nn.Module):
    def __init__(self, in_channels, out_channels, k_rank, modes, expansion_factor, activation=F.relu):
        super().__init__()
        
        self.skip_connection_layer = nn.Conv1d(in_channels * expansion_factor, in_channels, 1)
        
        self.channels = nn.ModuleList()
        self.channels.append(PDIAE_IdentityChannel(k_rank, modes))
        self.channels.append(PDIAE_FourierChannel(k_rank, modes))
        
        if IDENTITY_MAP: # Architecture if including identity map.
            self.mlp_channels = len(self.channels) + 1
            self.identity_conv = nn.Conv1d(in_channels, out_channels, SKIP_CONNECTION_KERNEL_SIZE)
        else:
            self.mlp_channels = len(self.channels)
        
        self.in_ln = nn.LayerNorm(self.mlp_channels * out_channels) # Layer Norm
        
        self.channel_merger_linear = nn.Linear(self.mlp_channels * out_channels, out_channels)
        self.channel_merger_nonlinear = nn.Sequential(
            nn.Linear(self.mlp_channels * out_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_PROB),
            nn.Linear(2 * out_channels, out_channels),
            nn.Dropout(p=DROPOUT_PROB)
        )
        
        self.out_ln = nn.LayerNorm(out_channels)
        
        self.activation = activation
        
    def forward(self, x_list: list[Tensor]):
        x: Tensor = torch.cat(x_list, dim=1)
        x_shape = x.shape # (b, (block num)*c, s1, s2)
        
        # Skip connection layer.
        x = x.reshape(x_shape[0], x_shape[1], -1) # (b, (block num)*c, s1*s2)
        x = self.skip_connection_layer(x)
        x = x.reshape(x_shape[0], -1, x_shape[2], x_shape[3]) # (b, c, s1, s2)
        
        # Run input through parallel channels.
        x_channels = []
        for channel in self.channels:
            x_channels.append(channel(x)) # list containing (b, c', s1, s2) where c' is out channel
            
        if IDENTITY_MAP:
            x_shape = x.shape # (b, c, s1, s2)
            x = x.reshape(x.shape[0], x.shape[1], -1) # (b, c, s1*s2)
            x_identity = self.identity_conv(x) # (b, c, s1*s2) -> (b, c', s1*s2)
            x_identity = x_identity.reshape(x_shape[0], -1, x_shape[2], x_shape[3]) # (b, c, s1, s2)
            x_channels.append(x_identity)
        
        # Concatenate channel outputs.
        x = torch.cat(x_channels, dim=1) # (b, k*c, s1, s2) where k is (self.mlp_channels)
        x = x.movedim(1, -1) # (b, s1, s2, k*c)
        
        # Layernorm before merging channel outputs.
        x = self.in_ln(x) # (b, s1, s2, k*c)
        
        # Merge channel outputs.
        x = self.channel_merger_linear(x) + self.channel_merger_nonlinear(x) # (b, s1, s2, c) where c is output channel
        
        # Layernorm after merging channel outputs.
        x = self.out_ln(x) # (b, s1, s2, c)
        x = x.movedim(-1, 1) # (b, c, s1, s2)
        
        # Apply activation function.
        if self.activation is not None:
            x = self.activation(x)
        
        # Append output to input list. 
        x_list.append(x)
        return x_list
        
class PDIAE_IdentityChannel(nn.Module):
    def __init__(self, k_rank: int, modes: int):
        super().__init__()
        
        self.encoder = PDIAE_Encoder(k_rank, modes)
        self.fixed_model = ModesFNN(modes)
        self.decoder = PDIAE_Decoder(k_rank, modes)
        
    def forward(self, x: Tensor):
        target_size = (x.shape[2], x.shape[3])
        x = self.encoder(x)
        x = self.fixed_model(x)
        x = self.decoder(x, target_size=target_size)
        
        return x


class PDIAE_FourierChannel(PDIAE_IdentityChannel):
    def __init__(self, k_rank, modes):
        super().__init__(k_rank, modes)

    def forward(self, x: Tensor):
        
        # Fourier Channel Transform
        x_shape = x.shape # real(b, c, s1, s2)
        x = torch.fft.rfft2(x, norm='ortho') # complex(b, c, s1, s2/2)
        x = torch.view_as_real(x).reshape(x.shape[0], x.shape[1], x.shape[2], 2*x.shape[3]) # real(b, c, s1, ~s2)
        
        # Encoder -> Fixed Model -> Decoder
        x = super().forward(x) # real(b, c, s1, ~s2)
        
        # Fourier Channel Inverse Transform
        x = torch.view_as_complex(x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2, 2)) # complex(b, c, s1, s2/2)
        x = torch.fft.irfft2(x, s=[x_shape[2], x_shape[3]], norm="ortho") # real(b, c, s1, s2)
        
        return x
        

class PDIAE_Encoder(nn.Module):
    def __init__(self, k_rank: int, modes: int):
        super().__init__()
        
        self.k_rank = k_rank
        self.modes = modes
        
        self.spatial_net = KRankNets(k_rank)
        self.freq_net = KRankNets(k_rank)
        
    def forward(self, x: Tensor):
        
        # Pointwise product in spatial domain
        x_shape = x.shape # (b, c, s1, s2)
        x = x.unsqueeze(-1).repeat(1, 1, 1, 1, self.k_rank) # real(b, c, s1, s2, k)
        spatial = spatial_grid2([x_shape[2], x_shape[3]], x.device)
        x = x * self.spatial_net(spatial) # (s1, s2, 2) -> (s1, s2, k) * (b, c, s1, s2, k)
        
        # Fourier Transform
        x = torch.fft.rfft2(x, norm='ortho', dim=[2, 3]) # complex(b, c, s1, s2/2, k)
        
        # Pointwise product in frequency domain
        trunc_x = torch.zeros_like(x, dtype=torch.cfloat, device=x.device) # complex(b, c, s1, s2/2, k)
        freq = freq_grid2([x_shape[2], x_shape[3]], x.device) # (s1, s2/2, 2)

        trunc_x = rep_modes(trunc_x, self.modes, [2, 3])
        trunc_x[..., :self.modes, :self.modes, :] = rep_modes(x, self.modes, [2, 3])[..., :self.modes, :self.modes, :] * self.freq_net(rep_modes(freq, self.modes, [0, 1])[:self.modes, :self.modes, :]) # (m, m, 2) -> (m, m, k) * (b, c, m, m, k)
        trunc_x[..., :self.modes, -self.modes:, :] = rep_modes(x, self.modes, [2, 3])[..., :self.modes, -self.modes:, :] * self.freq_net(rep_modes(freq, self.modes, [0, 1])[:self.modes, -self.modes:, :]) # (m, m, 2) -> (m, m, k) * (b, c, m, m, k)
        
        # Sum across k axis
        trunc_x = trunc_x.sum(-1) # complex(b, c, s1, s2/2)
        
        return trunc_x
        
def rep_modes(x: Tensor, modes: int, axes: list[int]):
    repetitions = [1] * x.ndim
    for ax in axes:
        if modes > x.shape[ax]:
            repetitions[ax] = modes // x.shape[ax] + 1
    return x.repeat(*repetitions)

class PDIAE_Decoder(nn.Module):
    def __init__(self, k_rank: int, modes: int):
        super().__init__()
        
        self.k_rank = k_rank
        self.modes = modes
        
        self.spatial_net = KRankNets(k_rank)
        self.freq_net = KRankNets(k_rank)
        
    def forward(self, trunc_x: Tensor, target_size: tuple[int, int]):
        
        # Pointwise product in frequency domain
        trunc_x = trunc_x.unsqueeze(-1).repeat(1, 1, 1, 1, self.k_rank) # (b, c, s1, s2/2, k)
        freq = freq_grid2([target_size[0], target_size[1]], trunc_x.device) # (s1, s2/2, 2)
        x: Tensor = torch.zeros_like(trunc_x, dtype=torch.cfloat, device=trunc_x.device)

        x[..., :self.modes, :self.modes, :] = trunc_x[..., :self.modes, :self.modes, :] * self.freq_net(rep_modes(freq, self.modes, [0, 1])[:self.modes, :self.modes, :])  # (m, m, 2) -> (m, m, k) * (b, c, m, m, k)
        x[..., -self.modes:, :self.modes, :] = trunc_x[..., -self.modes:, :self.modes, :] * self.freq_net(rep_modes(freq, self.modes, [0, 1])[-self.modes:, :self.modes, :]) # (m, m, 2) -> (m, m, k) * (b, c, m, m, k)
        
        # Inverse Fourier Transform
        x = torch.fft.irfft2(x, s=target_size, norm="ortho", dim=[2, 3]) # (b, c, s1, s2, k)
        
        # Pointwise product in spatial domain
        spatial = spatial_grid2([target_size[0], target_size[1]], x.device)
        x = x * self.spatial_net(spatial) # (s1, s2, 2) -> (s1, s2, k) * (b, c, s1, s2, k)
        
        # Sum across k axis
        x = x.sum(-1)
        
        return x
    
KRANK_HIDDENLAYERS = [64]
class KRankNets(nn.Module):
    def __init__(self, k_rank: int):
        super().__init__()
        
        self.k_rank = k_rank
        
        layer_dims = [DIM] + KRANK_HIDDENLAYERS + [1]
        self.nets = nn.ModuleList()
        for i in range(k_rank):
            net = nn.Sequential()
            for j in range(len(layer_dims) - 1):
                net.append(nn.Linear(layer_dims[j], layer_dims[j+1]))
                net.append(nn.ReLU())
            self.nets.append(net)
            
    def forward(self, grid: Tensor):
        """

        Parameters
        ----------
        grid : Tensor of shape (s1, s2, DIM)
            _description_

        Returns
        -------
        _type_
            _description_
        """
        
        grid_shape = grid.shape
        out = torch.zeros(grid_shape[0], grid_shape[1], self.k_rank, device=grid.device) # (s1, s2, k)
        
        for i in range(self.k_rank):
            out[:, :, [i]] = self.nets[i](grid) # (s1, s2, 2) -> (s1, s2, 1)
        
        return grid
        

def spatial_grid2(shape: list[int], device: str):
    s1 = torch.linspace(0, 1, shape[0], device=device).unsqueeze(-1).repeat(1, shape[1]) # s1 coords (s1, s2)
    s2 = torch.linspace(0, 1, shape[1], device=device).unsqueeze(0).repeat(shape[0], 1) # s2 coords (s1, s2)
    grid = torch.stack([s1, s2], dim=-1)
    return grid

def freq_grid2(shape: list[int], device: str):
    xi1 = torch.fft.fftfreq(shape[0], device=device).unsqueeze(-1).repeat(1, shape[1] // 2 + 1)
    xi2 = torch.fft.rfftfreq(shape[1], device=device).unsqueeze(0).repeat(shape[0], 1)
    freq = torch.stack([xi1, xi2], dim=-1)
    return freq


class ModesFNN(nn.Module):
    def __init__(self, modes: int):
        super().__init__()
        
        self.modes = modes
        self.fixed_model1 = FNN(modes)
        self.fixed_model2 = FNN(modes)
    
    def forward(self, x: Tensor):
        out = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        out[..., :self.modes, :self.modes] = self.fixed_model1(x[..., :self.modes, :self.modes])
        out[..., -self.modes:, :self.modes] = self.fixed_model2(x[..., -self.modes:, :self.modes])
        return out


FNN_HIDDEN = [64]
class FNN(nn.Module):
    def __init__(self, modes):
        super().__init__()
        
        layer_dims = [modes**2] + FNN_HIDDEN + [modes**2]
        
        self.linears = nn.ModuleList()
        for j in range(len(layer_dims) - 1):
            self.linears.append(
                nn.Linear(layer_dims[j], layer_dims[j+1], dtype=torch.cfloat)
            )

    def forward(self, x: Tensor):
        x_shape = x.shape # (b, c, m, m)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (b, c, m^2)
        for linear in self.linears:
            x = linear(x)
            x = F.relu(x.real) + 1.0j * F.relu(x.imag)
        x = x.reshape(x.shape[0], x.shape[1], x_shape[2], x_shape[3]) # complex(b, c, m, m)
        return x