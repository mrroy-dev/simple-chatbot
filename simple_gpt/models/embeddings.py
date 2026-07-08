import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.1,
        weight_tying: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, offset: int = 0):
        x = self.token_embedding(idx)
        x = self.drop(x)
        return x
