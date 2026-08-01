import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ff_dim, bias=False)
        self.up = nn.Linear(d_model, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim,  d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(x) * F.silu(self.gate(x)))
