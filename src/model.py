import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.generate import generate

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

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ff_dim, bias=False)
        self.up = nn.Linear(d_model, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim,  d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(x) * F.silu(self.gate(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)
        self.ffn = SwiGLU(d_model, ff_dim)
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.mha(self.norm1(x), pad_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def prefill(self, x):
        attn, kv = self.mha.prefill(self.norm1(x))
        x = x + self.drop(attn)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, list(kv)

    def forward_with_cache(self, x, kv, cache_len: int):
        x = x + self.drop(self.mha.forward_with_cache(self.norm1(x), kv, cache_len))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, ff_dim: int, max_seq_len: int, dropout: float, pad_token_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        std = self.d_model ** -0.5
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        pad_mask = attention_mask.bool() if attention_mask is not None \
                   else (input_ids != self.pad_token_id)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.lm_head(self.norm(x))

    def init_cache(self, batch_size: int, max_gen_len: int, device: torch.device):
        d_k = self.d_model // self.num_heads
        return [
            [
                torch.empty(batch_size, self.num_heads, max_gen_len, d_k, device=device),
                torch.empty(batch_size, self.num_heads, max_gen_len, d_k, device=device),
            ]
            for _ in self.blocks
        ]

    def prefill(self, input_ids: torch.Tensor, kv_cache=None):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block.prefill(x)
            if kv_cache is not None:
                kv_cache[i][0][:B, :, :T, :] = kv[0]
                kv_cache[i][1][:B, :, :T, :] = kv[1]
                new_cache.append(kv_cache[i])
            else:
                new_cache.append(kv)
        logits = self.lm_head(self.norm(x))[:, -1, :]
        return logits, new_cache

    def decode_step(self, token_ids: torch.Tensor, kv_cache, cache_len: int):
        token_ids = token_ids.view(-1, 1)
        x = self.embed(token_ids)
        for block, kv in zip(self.blocks, kv_cache):
            x = block.forward_with_cache(x, kv, cache_len)
        return self.lm_head(self.norm(x))[:, 0, :]

    def generate_response(self, user_input, tokenizer, **kwargs):
        return generate(self, user_input, tokenizer, **kwargs)
