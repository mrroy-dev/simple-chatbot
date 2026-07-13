import torch
import torch.nn as nn
from typing import Optional, List, Iterator
from simple_gpt.inference.sampling import sample


class Generator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.kv_caches = None

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[List[int]] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        add_bos: bool = True,
        add_user_bot: bool = True,
        stream: bool = False,
    ) -> str:
        if input_ids is not None:
            ids = list(input_ids)
        elif prompt is not None:
            ids = self.tokenizer.encode(prompt)
            if add_bos:
                ids = [self.tokenizer.bos_id()] + ids
            if add_user_bot:
                ids = ids + [self.tokenizer.bot_id()]
        else:
            raise ValueError("Either prompt or input_ids must be provided")

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        if stream:
            return self._stream_generate(
                idx, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty, eos_token_id, stop_tokens,
            )

        self.kv_caches = [None] * self.model.config.n_layer
        prompt_len = idx.size(1)

        for _ in range(max_new_tokens):
            if idx.size(1) > self.model.config.block_size:
                idx = idx[:, -self.model.config.block_size:]
                self.kv_caches = [None] * self.model.config.n_layer
                prompt_len = idx.size(1)

            logits, _, new_kv = self.model(
                idx[:, -1:] if self.kv_caches[0] is not None else idx,
                kv_caches=self.kv_caches,
                offset=0 if self.kv_caches[0] is None else idx.size(1) - 1,
            )
            self.kv_caches = new_kv
            logits = logits[:, -1, :]

            next_id = sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                input_ids=idx,
            )

            idx = torch.cat([idx, next_id], dim=1)

            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
            if stop_tokens is not None and next_id.item() in stop_tokens:
                break

        generated_ids = idx[0, prompt_len:].tolist()
        if eos_token_id is not None and eos_token_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_token_id)]

        return self.tokenizer.decode(generated_ids, skip_special=True)

    def _stream_generate(self, idx, max_new_tokens, temperature, top_k, top_p,
                          repetition_penalty, eos_token_id, stop_tokens):
        self.kv_caches = [None] * self.model.config.n_layer
        prompt_len = idx.size(1)
        generated_ids = []

        for _ in range(max_new_tokens):
            if idx.size(1) > self.model.config.block_size:
                idx = idx[:, -self.model.config.block_size:]
                self.kv_caches = [None] * self.model.config.n_layer
                prompt_len = idx.size(1)

            logits, _, new_kv = self.model(
                idx[:, -1:] if self.kv_caches[0] is not None else idx,
                kv_caches=self.kv_caches,
                offset=0 if self.kv_caches[0] is None else idx.size(1) - 1,
            )
            self.kv_caches = new_kv
            logits = logits[:, -1, :]

            next_id = sample(
                logits, temperature, top_k, top_p, repetition_penalty, idx
            )

            idx = torch.cat([idx, next_id], dim=1)
            generated_ids.append(next_id.item())

            yield self.tokenizer.decode(generated_ids, skip_special=True)

            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
            if stop_tokens is not None and next_id.item() in stop_tokens:
                break

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(
                self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            )
        return results

    def reset_kv_cache(self):
        self.kv_caches = None
