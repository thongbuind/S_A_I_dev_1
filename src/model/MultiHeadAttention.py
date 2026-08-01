import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.RotaryPositionalEmbedding import RotaryPositionalEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)

    def _project_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        return q, k, v

    def _merge(self, out: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = out.shape
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k))

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        pos = torch.arange(T, device=x.device)
        q = self.rope.apply_rope(q, pos)
        k = self.rope.apply_rope(k, pos)

        dropout_p = self.dropout_rate if self.training else 0.0

        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn_mask = causal[None, None, :, :]

        if pad_mask is not None:
            pad = torch.zeros(B, 1, 1, T, device=x.device)
            pad.masked_fill_(~pad_mask[:, None, None, :], float('-inf'))
            attn_mask = attn_mask + pad

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=dropout_p,
        )
        return self._merge(out)

    def prefill(self, x: torch.Tensor):
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        pos = torch.arange(T, device=x.device)
        q = self.rope.apply_rope(q, pos)
        k = self.rope.apply_rope(k, pos)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self._merge(out), (k, v)

    def forward_with_cache(self, x: torch.Tensor, past_kv, cache_len: int):
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        pos = torch.arange(cache_len, cache_len + T, device=x.device)
        q = self.rope.apply_rope(q, pos)
        k = self.rope.apply_rope(k, pos)

        past_kv[0][:B, :, cache_len:cache_len + T, :] = k
        past_kv[1][:B, :, cache_len:cache_len + T, :] = v

        k_full = past_kv[0][:B, :, :cache_len + T, :]
        v_full = past_kv[1][:B, :, :cache_len + T, :]

        out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        return self._merge(out)
