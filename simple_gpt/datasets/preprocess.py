import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from simple_gpt.datasets.cleaners import TextCleaner
from simple_gpt.datasets.filters import DataFilter
from simple_gpt.datasets.conversation_builder import ConversationBuilder


class Preprocessor:
    def __init__(self, tokenizer=None, seq_len: int = 512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.cleaner = TextCleaner()
        self.filter = DataFilter(tokenizer)
        self.builder = ConversationBuilder(tokenizer, seq_len)

    def process(
        self,
        input_path: str,
        output_path: str,
        format: str = "auto",
        val_split: float = 0.05,
        sliding_window: int = 4,
        clean_kwargs: Optional[dict] = None,
    ) -> Tuple[str, str]:
        examples = self.builder.load_and_build(input_path, format=format, sliding_window=sliding_window)
        print(f"Loaded {len(examples)} raw examples from {input_path}")

        clean_kwargs = clean_kwargs or {}
        for ex in examples:
            ex["context"] = self.cleaner.clean(ex["context"], **clean_kwargs)
            ex["response"] = self.cleaner.clean(ex["response"], **clean_kwargs)

        examples = self.filter.filter_dataset(examples)
        print(f"After filtering: {len(examples)} examples")

        stats = self.filter.dataset_stats(examples)
        print(f"Stats: {stats}")

        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - val_split))
        train = examples[:split_idx]
        val = examples[split_idx:]

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        train_path = output_path.replace(".json", "_train.json")
        val_path = output_path.replace(".json", "_val.json")
        if not output_path.endswith(".json"):
            train_path = os.path.join(out_dir, "train.json")
            val_path = os.path.join(out_dir, "val.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train, f, ensure_ascii=False, indent=2)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(train)} train + {len(val)} val examples")
        return train_path, val_path
