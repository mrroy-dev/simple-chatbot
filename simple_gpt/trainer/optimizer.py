import torch
import torch.nn as nn
from typing import Iterable, Optional
from simple_gpt.config import OptimizerConfig, TrainConfig


def configure_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    opt_cfg = config.optimizer
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": opt_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    fused_available = hasattr(torch.optim, "AdamW") and opt_cfg.fused
    use_fused = fused_available and config.mixed_precision != "fp32"

    if opt_cfg.name == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=opt_cfg.lr,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            eps=opt_cfg.eps,
            fused=use_fused,
        )
    else:
        return torch.optim.AdamW(
            groups,
            lr=opt_cfg.lr,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            eps=opt_cfg.eps,
            fused=use_fused,
        )
