import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Dict, Any, Optional, Iterator, Tuple


def _build_supervised_example(
    item: Dict[str, Any],
    tokenizer,
    seq_len: int,
    add_bos: bool,
    add_eos: bool,
) -> Optional[Tuple[list, list]]:
    ctx = item.get("context", "")
    resp = item.get("response", "")
    ctx_ids = tokenizer.encode(ctx)
    resp_ids = tokenizer.encode(resp)

    if not resp_ids:
        return None

    bos = tokenizer.bos_id()
    eos = tokenizer.eos_id()
    user_t = tokenizer.user_id()
    bot_t = tokenizer.bot_id()
    pad = tokenizer.pad_id()

    prefix = []
    if add_bos:
        prefix.append(bos)
    prefix.append(user_t)
    prefix.extend(ctx_ids)
    prefix.append(bot_t)

    fixed_prefix_len = (1 if add_bos else 0) + 2
    eos_len = 1 if add_eos else 0
    response_budget = seq_len - fixed_prefix_len - eos_len
    if response_budget <= 0:
        return None

    if len(resp_ids) > response_budget:
        resp_ids = resp_ids[:response_budget]

    ctx_budget = seq_len - fixed_prefix_len - len(resp_ids) - eos_len
    if ctx_budget < 0:
        return None
    if len(ctx_ids) > ctx_budget:
        # Keep the most recent prompt tokens and always leave supervised response tokens visible.
        ctx_ids = ctx_ids[-ctx_budget:] if ctx_budget > 0 else []

    ids = []
    if add_bos:
        ids.append(bos)
    ids.append(user_t)
    ids.extend(ctx_ids)
    ids.append(bot_t)
    resp_start = len(ids)
    ids.extend(resp_ids)
    if add_eos:
        ids.append(eos)

    loss_mask = [0] * resp_start + [1] * (len(ids) - resp_start)
    pad_len = seq_len - len(ids)
    if pad_len < 0:
        return None

    input_ids = ids + [pad] * pad_len
    mask = loss_mask + [0] * pad_len

    if sum(mask[1:]) == 0:
        return None
    return input_ids, mask


class ConversationDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        seq_len: int = 512,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.examples = []
        pad = tokenizer.pad_id()
        self.pad_id = pad

        for item in data:
            example = _build_supervised_example(item, tokenizer, seq_len, add_bos, add_eos)
            if example is not None:
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, mask = self.examples[i]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.long)
        attn = torch.tensor([1 if t != self.pad_id else 0 for t in ids[:-1]], dtype=torch.long)
        return x, y, m, attn


class StreamingConversationDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int = 512,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __iter__(self) -> Iterator:
        pad = self.tokenizer.pad_id()

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                example = _build_supervised_example(
                    item,
                    self.tokenizer,
                    self.seq_len,
                    self.add_bos,
                    self.add_eos,
                )
                if example is None:
                    continue
                input_ids, mask = example

                x = torch.tensor(input_ids[:-1], dtype=torch.long)
                y = torch.tensor(input_ids[1:], dtype=torch.long)
                m = torch.tensor(mask[1:], dtype=torch.long)
                attn = torch.tensor([1 if t != pad else 0 for t in input_ids[:-1]], dtype=torch.long)
                yield x, y, m, attn


def ChatDataLoader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not isinstance(dataset, IterableDataset) else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )
