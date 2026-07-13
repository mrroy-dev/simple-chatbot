import re
from typing import List, Dict, Any, Callable, Optional


class DataFilter:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def min_tokens(self, text: str, min_tokens: int = 5) -> bool:
        if self.tokenizer:
            return len(self.tokenizer.encode(text)) >= min_tokens
        return len(text.split()) >= min_tokens

    def max_tokens(self, text: str, max_tokens: int = 2048) -> bool:
        if self.tokenizer:
            return len(self.tokenizer.encode(text)) <= max_tokens
        return len(text.split()) <= max_tokens

    def not_empty(self, text: str) -> bool:
        return bool(text.strip())

    def min_length(self, text: str, min_chars: int = 3) -> bool:
        return len(text.strip()) >= min_chars

    def max_length(self, text: str, max_chars: int = 10000) -> bool:
        return len(text.strip()) <= max_chars

    def no_urls_only(self, text: str) -> bool:
        cleaned = re.sub(r"https?://[^\s]+", "", text).strip()
        return len(cleaned) > 0

    def min_words(self, text: str, min_words: int = 2) -> bool:
        return len(text.split()) >= min_words

    def filter_pair(self, pair: Dict[str, Any], filters: List[Callable] = None) -> bool:
        if filters is None:
            filters = [
                lambda p: self.not_empty(p.get("context", "")),
                lambda p: self.not_empty(p.get("response", "")),
                lambda p: self.min_tokens(p.get("context", "")),
                lambda p: self.min_tokens(p.get("response", "")),
            ]
        return all(f(pair) for f in filters)

    def filter_dataset(self, data: List[Dict[str, Any]], filters: Optional[List[Callable]] = None) -> List[Dict[str, Any]]:
        return [d for d in data if self.filter_pair(d, filters)]

    def dataset_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_lens = []
        response_lens = []
        total_tokens = 0
        for d in data:
            ctx = d.get("context", "")
            resp = d.get("response", "")
            if self.tokenizer:
                context_lens.append(len(self.tokenizer.encode(ctx)))
                response_lens.append(len(self.tokenizer.encode(resp)))
                total_tokens += len(self.tokenizer.encode(ctx)) + len(self.tokenizer.encode(resp))
            else:
                context_lens.append(len(ctx.split()))
                response_lens.append(len(resp.split()))
                total_tokens += len(ctx.split()) + len(resp.split())
        return {
            "total_examples": len(data),
            "total_tokens": total_tokens,
            "avg_context_tokens": sum(context_lens) / max(len(context_lens), 1),
            "avg_response_tokens": sum(response_lens) / max(len(response_lens), 1),
            "max_context_tokens": max(context_lens) if context_lens else 0,
            "max_response_tokens": max(response_lens) if response_lens else 0,
        }

    def lang_filter(self, text: str, target_lang: str = "en") -> bool:
        try:
            import langdetect
            return langdetect.detect(text) == target_lang
        except ImportError:
            return True
        except langdetect.lang_detect_exception.LangDetectException:
            return True
