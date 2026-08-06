import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.ff_dim = ff_dim
        self.gate_up = nn.Linear(d_model, 2 * ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(up * F.silu(gate))
