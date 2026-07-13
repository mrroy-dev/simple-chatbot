import math
from typing import List


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 100.0))


def token_accuracy(logits, targets, loss_mask=None):
    preds = logits.argmax(dim=-1)
    correct = preds == targets
    if loss_mask is not None:
        correct = correct & loss_mask.bool()
        total = loss_mask.sum().item()
    else:
        total = targets.numel()
    return correct.sum().item() / max(total, 1)


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not cand_tokens:
        return 0.0
    precisions = []
    for n in range(1, min(max_n, len(cand_tokens)) + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        cand_ngrams = _get_ngrams(cand_tokens, n)
        matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
        total = len(cand_ngrams)
        precisions.append(matches / max(total, 1))
    if not precisions:
        return 0.0
    geometric = math.exp(sum(math.log(max(p, 1e-10)) for p in precisions) / len(precisions))
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
    return bp * geometric


def _get_ngrams(tokens: List[str], n: int) -> set:
    return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_1_f(reference: str, candidate: str) -> float:
    ref_tokens = set(reference.split())
    cand_tokens = set(candidate.split())
    if not cand_tokens:
        return 0.0
    overlap = ref_tokens & cand_tokens
    precision = len(overlap) / max(len(cand_tokens), 1)
    recall = len(overlap) / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def throughput(tokens: int, time_seconds: float) -> float:
    return tokens / max(time_seconds, 1e-6)
