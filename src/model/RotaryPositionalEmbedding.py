import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim phải chẵn"
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq) # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1) # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[positions][None, None, :, :]
        sin = self.sin_cached[positions][None, None, :, :]
        return x * cos + self._rotate_half(x) * sin
