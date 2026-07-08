import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List
from simple_gpt.utils.metrics import perplexity, token_accuracy


class Evaluator:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        total_tokens = 0

        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            x, y, m, attn = batch
            x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)

            logits, loss, _ = self.model(x, targets=y, loss_mask=m)
            total_loss += loss.item()
            total_acc += token_accuracy(logits, y, m)
            total_tokens += m.sum().item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)

        self.model.train()
        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity(avg_loss),
            "val_accuracy": avg_acc,
            "val_tokens": total_tokens,
        }

    @torch.no_grad()
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        results = self.evaluate(dataloader)
        return results["val_perplexity"]

    @torch.no_grad()
    def compute_loss(self, dataloader: DataLoader) -> float:
        results = self.evaluate(dataloader)
        return results["val_loss"]
