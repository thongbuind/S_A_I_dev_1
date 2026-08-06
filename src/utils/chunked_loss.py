import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def chunked_lm_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                     token_weight: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    N = hidden.shape[0]
    total_loss = hidden.new_zeros(())

    def _chunk_fn(h_chunk, t_chunk, w_chunk):
        logits_chunk = F.linear(h_chunk, weight).float()
        loss_per_token = F.cross_entropy(logits_chunk, t_chunk, reduction='none')
        return (loss_per_token * w_chunk).sum()

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_loss = checkpoint(
            _chunk_fn,
            hidden[start:end],
            targets[start:end],
            token_weight[start:end],
            use_reentrant=False,
        )
        total_loss = total_loss + chunk_loss

    return total_loss
