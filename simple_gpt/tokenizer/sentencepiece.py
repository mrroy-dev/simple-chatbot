from typing import List, Optional, Dict
from simple_gpt.tokenizer.tokenizer import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 8000):
        super().__init__()
        self.vocab_size = vocab_size
        self._sp = None
        if model_path:
            self.load(model_path)

    def _build_vocab(self, texts: List[str], min_freq: int, max_vocab_size: int):
        import sentencepiece as spm
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in texts:
                f.write(text + "\n")
            corpus_path = f.name

        model_path = tempfile.mktemp(suffix=".model")
        specials = list(self.SPECIAL_TOKENS.values())
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_path.replace(".model", ""),
            vocab_size=min(max_vocab_size, 32000),
            model_type="bpe",
            character_coverage=1.0,
            control_symbols=specials,
            user_defined_symbols=[],
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        os.unlink(corpus_path)
        self._sp = spm.SentencePieceProcessor(model_file=model_path + ".model")
        self.vocab = {self._sp.IdToPiece(i): i for i in range(self._sp.GetPieceSize())}
        self.inv_vocab = {i: self._sp.IdToPiece(i) for i in range(self._sp.GetPieceSize())}
        for name, token in self.SPECIAL_TOKENS.items():
            tid = self._sp.PieceToId(token)
            if tid != self._sp.unk_id():
                self.special_ids[name] = tid

    def encode(self, text: str) -> List[int]:
        if self._sp is None:
            return []
        return self._sp.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self._sp is None:
            return ""
        return self._sp.decode(ids)

    def save(self, path: str):
        import shutil
        if self._sp is not None:
            shutil.copy(self._sp.model_file, path)

    def load(self, path: str):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor(model_file=path)
        self.vocab = {self._sp.IdToPiece(i): i for i in range(self._sp.GetPieceSize())}
        self.inv_vocab = {i: self._sp.IdToPiece(i) for i in range(self._sp.GetPieceSize())}
        for name, token in self.SPECIAL_TOKENS.items():
            tid = self._sp.PieceToId(token)
            if tid != self._sp.unk_id():
                self.special_ids[name] = tid

    def __len__(self) -> int:
        return self._sp.GetPieceSize() if self._sp else 0
