import math
import torch
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional
from simple_gpt.config import SchedulerConfig


class WarmupCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            lr_scale = step / max(self.warmup_steps, 1)
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]


class WarmupLinearScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            lr_scale = step / max(self.warmup_steps, 1)
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        linear_decay = 1.0 - progress
        return [
            self.min_lr + (base_lr - self.min_lr) * linear_decay
            for base_lr in self.base_lrs
        ]


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    total_steps: int,
) -> LRScheduler:
    schedulers = {
        "cosine": WarmupCosineScheduler,
        "linear": WarmupLinearScheduler,
    }
    cls = schedulers.get(config.name, WarmupCosineScheduler)
    return cls(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr=config.min_lr,
    )
