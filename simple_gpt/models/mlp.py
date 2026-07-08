import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, n_embd: int, hidden_mult: int = 4, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(2 * n_embd * hidden_mult / 3)
        hidden_dim = _round_up_to_multiple(hidden_dim, 64)
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return self.dropout(x)


class GeGLUMLP(nn.Module):
    def __init__(self, n_embd: int, hidden_mult: int = 4, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(2 * n_embd * hidden_mult / 3)
        hidden_dim = _round_up_to_multiple(hidden_dim, 64)
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return self.dropout(x)


def _round_up_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple
