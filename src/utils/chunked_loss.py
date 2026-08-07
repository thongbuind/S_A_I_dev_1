import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def chunked_lm_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                     token_weight: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    N = hidden.shape[0]
    total_loss = hidden.new_zeros((), dtype=torch.float32)

    def _chunk_fn(h_chunk, t_chunk, w_chunk):
        logits_chunk = F.linear(h_chunk, weight).float()
        loss_per_token = F.cross_entropy(logits_chunk, t_chunk, reduction='none')
        return (loss_per_token * w_chunk).sum()

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_args = (
            hidden[start:end],
            targets[start:end],
            token_weight[start:end],
        )
        if torch.is_grad_enabled():
            chunk_loss = checkpoint(
                _chunk_fn, *chunk_args, use_reentrant=False,
            )
        else:
            chunk_loss = _chunk_fn(*chunk_args)
        total_loss = total_loss + chunk_loss

    return total_loss

def chunked_sft_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    token_weight: torch.Tensor,
    penalty_engine=None,
    chunk_size: int = 4096,
):
    """Tính CE + penalty theo chunk mà không vật lý hóa full logits.

    PenaltyEngine trả về mean theo loss mask. Nhân lại với số token
    hợp lệ trong từng chunk rồi chia một lần ở caller giữ nguyên
    công thức mean trên toàn batch.
    """
    N = hidden.shape[0]
    total_ce = hidden.new_zeros((), dtype=torch.float32)
    total_penalty = hidden.new_zeros((), dtype=torch.float32)

    def _chunk_fn(h_chunk, i_chunk, t_chunk, w_chunk):
        logits_chunk = F.linear(h_chunk, weight).float()
        loss_per_token = F.cross_entropy(
            logits_chunk, t_chunk, reduction="none"
        )
        ce_sum = (loss_per_token * w_chunk).sum()

        if penalty_engine is None:
            penalty_sum = logits_chunk.new_zeros(())
        else:
            penalty_mean = penalty_engine(
                logits=logits_chunk.unsqueeze(0),
                inputs=i_chunk.unsqueeze(0),
                targets=t_chunk.unsqueeze(0),
                loss_mask=w_chunk.unsqueeze(0),
            )
            penalty_sum = penalty_mean * w_chunk.sum()

        return ce_sum, penalty_sum

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_args = (
            hidden[start:end],
            inputs[start:end],
            targets[start:end],
            token_weight[start:end],
        )
        if torch.is_grad_enabled():
            ce_sum, penalty_sum = checkpoint(
                _chunk_fn, *chunk_args, use_reentrant=False,
            )
        else:
            ce_sum, penalty_sum = _chunk_fn(*chunk_args)

        total_ce = total_ce + ce_sum
        total_penalty = total_penalty + penalty_sum

    return total_ce, total_penalty
