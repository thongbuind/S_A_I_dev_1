import torch
import torch.nn as nn
from src.inference.generate import generate
from src.model.DecoderBlock import DecoderBlock

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
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, num_kv_heads, ff_dim, max_seq_len, dropout)
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
                torch.empty(batch_size, self.num_kv_heads, max_gen_len, d_k, device=device),
                torch.empty(batch_size, self.num_kv_heads, max_gen_len, d_k, device=device),
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
