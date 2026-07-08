import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from simple_gpt.models.embeddings import GPTEmbeddings
from simple_gpt.models.transformer import TransformerBlock, RMSNorm
from simple_gpt.config import ModelConfig


class SimpleGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            block_size=config.block_size,
            dropout=config.embd_dropout,
            weight_tying=config.weight_tying,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                n_kv_head=config.n_kv_head,
                block_size=config.block_size,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                resid_dropout=config.resid_dropout,
                bias=config.bias,
                mlp_ratio=config.mlp_ratio,
                activation=config.activation,
                norm_eps=config.norm_eps,
                rope_base=config.rope_base,
            )
            for _ in range(config.n_layer)
        ])

        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.embeddings.token_embedding.weight

        self.apply(self._init_weights)
        self._post_init()

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _post_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and hasattr(module, "out_proj"):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std / math.sqrt(2 * self.config.n_layer))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[tuple]]] = None,
        offset: int = 0,
    ):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        x = self.embeddings(idx, offset=offset)

        new_kv_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, cache, offset)
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss_all = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            if loss_mask is not None:
                mask_flat = loss_mask.reshape(-1).float()
                loss = (loss_all * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            else:
                loss = loss_all.mean()

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
    ):
        kv_caches = [None] * self.config.n_layer
        generated = idx

        for _ in range(max_new_tokens):
            if generated.size(1) > self.block_size:
                generated = generated[:, -self.block_size:]
                kv_caches = [None] * self.config.n_layer
                offset = 0
            else:
                offset = 0

            logits, _, new_kv_caches = self.forward(
                generated[:, -1:] if kv_caches[0] is not None else generated,
                kv_caches=kv_caches,
                offset=0 if kv_caches[0] is None else generated.size(1) - 1,
            )
            kv_caches = new_kv_caches
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty is not None:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
            if stop_tokens is not None and next_id.item() in stop_tokens:
                break

        return generated

    @torch.no_grad()
    def generate_with_kv_cache(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ):
        return self.generate(
            idx=idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
        )

    def configure_optimizer(self, config):
        from simple_gpt.trainer.optimizer import configure_optimizer
        return configure_optimizer(self, config)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu"):
        import torch
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt.get("model_config") or ModelConfig(**ckpt["config"])
        if isinstance(cfg, dict):
            cfg = ModelConfig(**cfg)
        model = cls(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model, ckpt
