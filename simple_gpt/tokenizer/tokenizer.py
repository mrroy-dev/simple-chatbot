import abc
import json
from typing import List, Optional, Dict, Any


class BaseTokenizer(abc.ABC):
    SPECIAL_TOKENS = {
        "pad": "<pad>",
        "unk": "<unk>",
        "bos": "<bos>",
        "eos": "<eos>",
        "user": "<user>",
        "bot": "<bot>",
        "system": "<system>",
        "tool": "<tool>",
    }

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.special_ids: Dict[str, int] = {}

    @abc.abstractmethod
    def _build_vocab(self, texts: List[str], min_freq: int, max_vocab_size: int):
        ...

    @abc.abstractmethod
    def encode(self, text: str) -> List[int]:
        ...

    @abc.abstractmethod
    def decode(self, ids: List[int]) -> str:
        ...

    def __len__(self) -> int:
        return len(self.vocab)

    def pad_id(self) -> int:
        return self.special_ids.get("pad", 0)

    def unk_id(self) -> int:
        return self.special_ids.get("unk", 0)

    def bos_id(self) -> int:
        return self.special_ids.get("bos", 0)

    def eos_id(self) -> int:
        return self.special_ids.get("eos", 0)

    def user_id(self) -> int:
        return self.special_ids.get("user", 0)

    def bot_id(self) -> int:
        return self.special_ids.get("bot", 0)

    def system_id(self) -> int:
        return self.special_ids.get("system", 0)

    def tool_id(self) -> int:
        return self.special_ids.get("tool", 0)

    def encode_batch(
        self,
        texts: List[str],
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> Dict[str, Any]:
        encoded = [self.encode(t) for t in texts]
        if add_special_tokens:
            encoded = [[self.bos_id()] + ids + [self.eos_id()] for ids in encoded]
        if truncation and max_length:
            encoded = [ids[:max_length] for ids in encoded]
        max_len = max(len(ids) for ids in encoded) if padding else 0
        attn_mask = []
        if padding:
            for i, ids in enumerate(encoded):
                pad_len = max_len - len(ids)
                encoded[i] = ids + [self.pad_id()] * pad_len
                attn_mask.append([1] * (len(ids) - pad_len) + [0] * pad_len)
        else:
            attn_mask = [[1] * len(ids) for ids in encoded]
        return {
            "input_ids": encoded,
            "attention_mask": attn_mask,
        }

    def decode_batch(self, batch: List[List[int]], skip_special: bool = True) -> List[str]:
        return [self.decode(ids) for ids in batch]

    def save_vocab(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "special_ids": self.special_ids}, f, ensure_ascii=False)

    def load_vocab(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.special_ids = data["special_ids"]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def vocab_size(self) -> int:
        return len(self.vocab)

    def stats(self) -> dict:
        return {
            "vocab_size": len(self.vocab),
            "special_tokens": {k: v for k, v in self.special_ids.items()},
            "unk_id": self.unk_id(),
            "pad_id": self.pad_id(),
        }

    def coverage_report(self, texts: List[str]) -> dict:
        total, covered = 0, 0
        unk = self.unk_id()
        for t in texts:
            ids = self.encode(t)
            total += len(ids)
            covered += sum(1 for i in ids if i != unk)
        return {
            "total_tokens": total,
            "covered_tokens": covered,
            "coverage": covered / max(total, 1),
            "unk_ratio": (total - covered) / max(total, 1),
        }
