import torch
import torch.nn as nn
from src.inference.generate import generate
from src.model.DecoderBlock import DecoderBlock
from src.model.RotaryPositionalEmbedding import RotaryPositionalEmbedding

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_kv_heads: int, num_layers: int, ff_dim: int, max_seq_len: int, dropout: float, pad_token_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(d_k, max_seq_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, num_kv_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        causal = torch.triu(
            torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1
        )
        self.register_buffer("causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self):
        std = self.d_model ** -0.5
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_attn_mask(self, T: int, pad_mask, device):
        attn_mask = self.causal_mask[:T, :T][None, None, :, :]
        if pad_mask is not None:
            pad = torch.zeros(pad_mask.shape[0], 1, 1, T, device=device)
            pad.masked_fill_(~pad_mask[:, None, None, :], float('-inf'))
            attn_mask = attn_mask + pad
        return attn_mask

    def forward_features(self, input_ids: torch.Tensor, attention_mask=None, has_padding: bool = True) -> torch.Tensor:
        """Giống forward() nhưng DỪNG TRƯỚC lm_head — dùng cho training loss chunked
        (tránh vật lý hóa logits full (B*T, vocab_size))."""
        pad_mask = attention_mask.bool() if attention_mask is not None \
                   else (input_ids != self.pad_token_id)
        B, T = input_ids.shape
        x = self.embed(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        cos, sin = self.rope.get_cos_sin(pos)

        # Chỉ build mask (và cộng pad-bias) khi batch này thực sự có token PAD.
        # Nếu không có PAD, causal mask thuần == causal+pad mask về mặt toán học
        # (pad-bias toàn số 0) -> bỏ qua an toàn, không đổi kết quả.
        # has_padding được tính sẵn trên CPU trong collate_fn nên không tốn sync GPU ở đây.
        attn_mask = self._build_attn_mask(T, pad_mask, x.device) if has_padding else None

        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        return self.norm(x)  # (B, T, d_model) — CHƯA qua lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask=None, has_padding: bool = True) -> torch.Tensor:
        x = self.forward_features(input_ids, attention_mask, has_padding)
        return self.lm_head(x)

    def init_cache(self, batch_size: int, max_gen_len: int, device: torch.device):
        d_k = self.d_model // self.num_heads
        return [
            [
                torch.empty(batch_size, self.num_kv_heads, max_gen_len, d_k, device=device),
                torch.empty(batch_size, self.num_kv_heads, max_gen_len, d_k, device=device),
            ]
            for _ in self.blocks
        ]

    def prefill(self, input_ids: torch.Tensor, kv_cache=None):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        cos, sin = self.rope.get_cos_sin(pos)
        new_cache = []

        for i, block in enumerate(self.blocks):
            x, kv = block.prefill(x, cos, sin)
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
        pos = torch.arange(cache_len, cache_len + 1, device=token_ids.device)
        cos, sin = self.rope.get_cos_sin(pos)

        for block, kv in zip(self.blocks, kv_cache):
            x = block.forward_with_cache(x, kv, cache_len, cos, sin)
        return self.lm_head(self.norm(x))[:, 0, :]

    def generate_response(self, user_input, tokenizer, **kwargs):
        return generate(self, user_input, tokenizer, **kwargs)
