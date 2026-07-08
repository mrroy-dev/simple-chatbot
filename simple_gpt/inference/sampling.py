import torch
import torch.nn.functional as F
from typing import Optional


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = float("-inf")
    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float("-inf")
    return logits


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if repetition_penalty is not None and input_ids is not None:
        for token_id in set(input_ids[0].tolist()):
            logits[0, token_id] /= repetition_penalty

    if top_k is not None:
        logits = top_k_filtering(logits, top_k)
    if top_p is not None:
        logits = top_p_filtering(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
