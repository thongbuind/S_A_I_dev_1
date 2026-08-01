import torch.nn as nn
from src.model.SwiGLU import SwiGLU
from src.model.GroupedQueryAttention import GroupedQueryAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 ff_dim: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads, max_seq_len, dropout)
        self.ffn = SwiGLU(d_model, ff_dim)
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.gqa(self.norm1(x), pad_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def prefill(self, x):
        attn, kv = self.gqa.prefill(self.norm1(x))
        x = x + self.drop(attn)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, list(kv)

    def forward_with_cache(self, x, kv, cache_len: int):
        x = x + self.drop(self.gqa.forward_with_cache(self.norm1(x), kv, cache_len))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
