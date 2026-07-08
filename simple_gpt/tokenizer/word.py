import re
import json
from collections import Counter
from typing import List, Optional, Dict
from simple_gpt.tokenizer.tokenizer import BaseTokenizer


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class WordTokenizer(BaseTokenizer):
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        super().__init__()
        if vocab:
            self._load_vocab(vocab)

    def _load_vocab(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        for name, token in self.SPECIAL_TOKENS.items():
            if token in self.vocab:
                self.special_ids[name] = self.vocab[token]

    def _build_vocab(self, texts: List[str], min_freq: int = 1, max_vocab_size: int = 8000):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        vocab = {}
        for name, token in self.SPECIAL_TOKENS.items():
            vocab[token] = len(vocab)
            self.special_ids[name] = vocab[token]

        for tok, freq in counter.most_common():
            if freq < min_freq:
                continue
            if tok in vocab:
                continue
            if len(vocab) >= max_vocab_size:
                break
            vocab[tok] = len(vocab)

        self.vocab = vocab
        self.inv_vocab = {i: t for t, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        unk = self.unk_id()
        return [self.vocab.get(t, unk) for t in tokenize(text)]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            token = self.inv_vocab.get(i, "<unk>")
            if skip_special and token in self.SPECIAL_TOKENS.values():
                continue
            tokens.append(token)
        out = []
        for t in tokens:
            if out and re.match(r"^[^\w\s]$", t):
                out[-1] = out[-1] + t
            else:
                out.append(t)
        return " ".join(out)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "special_ids": self.special_ids,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.special_ids = data.get("special_ids", {})
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        return tok
