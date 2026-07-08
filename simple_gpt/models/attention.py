import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from simple_gpt.models.rotary import RotaryEmbedding, apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        block_size: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        if n_kv_head is None or n_kv_head > n_head:
            n_kv_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.n_rep = self.n_head // self.n_kv_head if self.n_kv_head > 0 else 1

        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        assert n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=block_size * 2, base=rope_base)

        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple] = None,
        offset: int = 0,
    ):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = self.rotary(q, offset=offset)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            new_kv = (k, v)
        else:
            new_kv = (k, v)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).flatten(1, 2)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).flatten(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        S = att.size(-1)
        att = att.masked_fill(self.mask[:, :, :T, :S] == 0, float("-inf"))
        att = F.softmax(att, dim=-1, dtype=torch.float32).to(x.dtype)
        att = self.attn_dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, new_kv
