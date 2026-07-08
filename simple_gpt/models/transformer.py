import torch
import torch.nn as nn
from typing import Optional
from simple_gpt.models.attention import GroupedQueryAttention
from simple_gpt.models.mlp import SwiGLUMLP, GeGLUMLP


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        block_size: int = 512,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = False,
        mlp_ratio: int = 4,
        activation: str = "swiglu",
        norm_eps: float = 1e-5,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.ln1 = RMSNorm(n_embd, eps=norm_eps)
        self.attn = GroupedQueryAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            dropout=attn_dropout,
            bias=bias,
            block_size=block_size,
            rope_base=rope_base,
        )
        self.ln2 = RMSNorm(n_embd, eps=norm_eps)
        mlp_cls = SwiGLUMLP if activation == "swiglu" else GeGLUMLP
        self.mlp = mlp_cls(
            n_embd=n_embd,
            hidden_mult=mlp_ratio,
            bias=bias,
            dropout=resid_dropout,
        )
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple] = None,
        offset: int = 0,
    ):
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache, offset)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv
