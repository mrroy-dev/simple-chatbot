import re
import json
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Tuple
from simple_gpt.tokenizer.tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        super().__init__()
        self.merges: Dict[Tuple[str, str], int] = {}
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\S+""")
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        if vocab:
            self._load_vocab(vocab)

    def _load_vocab(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        for name, token in self.SPECIAL_TOKENS.items():
            if token in self.vocab:
                self.special_ids[name] = self.vocab[token]

    def _build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab_size: int = 8000):
        token_freqs = Counter()
        for text in texts:
            words = self.pattern.findall(text)
            for word in words:
                encoded = tuple(bytes_to_ints(word.encode("utf-8")))
                token_freqs[encoded] += 1

        base_vocab = list(range(256))
        for name, token in self.SPECIAL_TOKENS.items():
            base_vocab.append(token)
        vocab = {i: bytes([i]) for i in range(256)}
        for i, t in enumerate(self.SPECIAL_TOKENS.values()):
            vocab[256 + i] = t

        tokens = {k: list(k) for k in token_freqs}
        num_merges = max_vocab_size - len(vocab)
        merges = {}

        def _bytes_to_str(b: bytes) -> str:
            return "".join(self.byte_encoder[x] for x in b)

        for i in range(num_merges):
            pair_counts = defaultdict(int)
            for word, freq in token_freqs.items():
                token_list = tokens.get(word)
                if not token_list or len(token_list) < 2:
                    continue
                for j in range(len(token_list) - 1):
                    pair = (token_list[j], token_list[j + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break
            most_common = max(pair_counts, key=pair_counts.get)
            merges[most_common] = len(vocab)
            new_id = len(vocab)

            left = vocab[most_common[0]]
            right = vocab[most_common[1]]
            left_str = _bytes_to_str(left) if isinstance(left, bytes) else left
            right_str = _bytes_to_str(right) if isinstance(right, bytes) else right
            vocab[new_id] = left_str + right_str

            for word in list(tokens.keys()):
                token_list = tokens[word]
                if len(token_list) < 2:
                    continue
                j = 0
                while j < len(token_list) - 1:
                    if (token_list[j], token_list[j + 1]) == most_common:
                        token_list = token_list[:j] + [new_id] + token_list[j + 2 :]
                    else:
                        j += 1
                tokens[word] = token_list

        self.merges = merges
        self.vocab = {}
        for token_id, token_bytes in vocab.items():
            if isinstance(token_bytes, bytes):
                self.vocab[_bytes_to_str(token_bytes)] = token_id
            else:
                self.vocab[token_bytes] = token_id
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        for name, token in self.SPECIAL_TOKENS.items():
            if token in self.vocab:
                self.special_ids[name] = self.vocab[token]

    def _bpe_encode(self, text: str) -> List[int]:
        words = self.pattern.findall(text)
        ids = []
        for word in words:
            chars = list(bytes_to_ints(word.encode("utf-8")))
            while len(chars) >= 2:
                min_pair = None
                min_rank = float("inf")
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    rank = self.merges.get(pair)
                    if rank is not None and rank < min_rank:
                        min_rank = rank
                        min_pair = pair
                if min_pair is None:
                    break
                a, b = min_pair
                merged_str = self.inv_vocab.get(a, self.byte_encoder.get(a, "")) + self.inv_vocab.get(b, self.byte_encoder.get(b, ""))
                merged_id = self.vocab.get(merged_str, a)
                chars = _replace_pair(chars, a, b, merged_id)
            ids.extend(chars)
        return ids

    def encode(self, text: str) -> List[int]:
        unk = self.unk_id()
        ids = self._bpe_encode(text)
        return [i if i in self.inv_vocab else unk for i in ids]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            token = self.inv_vocab.get(i, "\ufffd")
            if skip_special and token in self.SPECIAL_TOKENS.values():
                continue
            tokens.append(token)
        result = "".join(tokens)
        byte_seq = bytearray()
        for c in result:
            b = self.byte_decoder.get(c)
            if b is not None:
                byte_seq.append(b)
            else:
                byte_seq.extend(c.encode("utf-8"))
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
                "special_ids": self.special_ids,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.merges = {tuple(k.split(",")): v for k, v in data.get("merges", {}).items()}
        tok.special_ids = data.get("special_ids", {})
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        return tok

    def frequency_stats(self, texts: List[str]) -> dict:
        counter = Counter()
        for t in texts:
            counter.update(self.encode(t))
        return {
            "total": sum(counter.values()),
            "unique": len(counter),
            "most_common": counter.most_common(20),
            "oov_count": counter.get(self.unk_id(), 0),
        }


def bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) if c < 256 else chr(c) for b, c in zip(bs, cs)}


def bytes_to_ints(b: bytes) -> List[int]:
    return list(b)


def _replace_pair(chars: List[int], a: int, b: int, new_id: int) -> List[int]:
    result = []
    i = 0
    while i < len(chars):
        if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(chars[i])
            i += 1
    return result
