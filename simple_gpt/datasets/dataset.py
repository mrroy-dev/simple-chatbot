import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Dict, Any, Optional, Iterator


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
        bos = tokenizer.bos_id()
        eos = tokenizer.eos_id()
        user_t = tokenizer.user_id()
        bot_t = tokenizer.bot_id()
        pad = tokenizer.pad_id()

        for item in data:
            ctx = item.get("context", "")
            resp = item.get("response", "")
            ctx_ids = tokenizer.encode(ctx)
            resp_ids = tokenizer.encode(resp)

            ids = []
            if add_bos:
                ids.append(bos)
            ids.append(user_t)
            ids.extend(ctx_ids)
            ids.append(bot_t)
            ids.extend(resp_ids)
            if add_eos:
                ids.append(eos)

            if len(ids) > seq_len:
                ids = ids[:seq_len]

            resp_start = len(ctx_ids) + 2 + (1 if add_bos else 0)
            loss_mask = [0] * len(ids)
            for i in range(min(resp_start, len(ids)), len(ids)):
                loss_mask[i] = 1

            pad_len = seq_len - len(ids)
            input_ids = ids + [pad] * pad_len
            mask = loss_mask + [0] * pad_len

            self.examples.append((input_ids, mask))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, mask = self.examples[i]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.long)
        attn = torch.tensor([1 if t != 0 else 0 for t in ids[:-1]], dtype=torch.long)
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
        bos = self.tokenizer.bos_id()
        eos = self.tokenizer.eos_id()
        user_t = self.tokenizer.user_id()
        bot_t = self.tokenizer.bot_id()
        pad = self.tokenizer.pad_id()

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ctx = item.get("context", "")
                resp = item.get("response", "")
                ctx_ids = self.tokenizer.encode(ctx)
                resp_ids = self.tokenizer.encode(resp)

                ids = []
                if self.add_bos:
                    ids.append(bos)
                ids.append(user_t)
                ids.extend(ctx_ids)
                ids.append(bot_t)
                ids.extend(resp_ids)
                if self.add_eos:
                    ids.append(eos)

                if len(ids) > self.seq_len:
                    ids = ids[:self.seq_len]

                resp_start = len(ctx_ids) + 2 + (1 if self.add_bos else 0)
                loss_mask = [0] * len(ids)
                for i in range(min(resp_start, len(ids)), len(ids)):
                    loss_mask[i] = 1

                pad_len = self.seq_len - len(ids)
                input_ids = ids + [pad] * pad_len
                mask = loss_mask + [0] * pad_len

                x = torch.tensor(input_ids[:-1], dtype=torch.long)
                y = torch.tensor(input_ids[1:], dtype=torch.long)
                m = torch.tensor(mask[1:], dtype=torch.long)
                attn = torch.tensor([1 if t != 0 else 0 for t in input_ids[:-1]], dtype=torch.long)
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
