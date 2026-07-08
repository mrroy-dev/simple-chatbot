import json
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path


class ConversationBuilder:
    SPECIAL_TOKENS = {
        "bos": "<bos>",
        "eos": "<eos>",
        "user": "<user>",
        "bot": "<bot>",
        "system": "<system>",
        "tool": "<tool>",
    }

    def __init__(self, tokenizer=None, seq_len: int = 512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def load_json(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def load_txt(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        conversations = []
        for block in content.strip().split("\n\n"):
            turns = [t.strip() for t in block.split("\n") if t.strip()]
            if len(turns) >= 2:
                conv = []
                for i, turn in enumerate(turns):
                    role = "user" if i % 2 == 0 else "bot"
                    conv.append({"role": role, "content": turn})
                conversations.append({"conversation": conv})
        return conversations

    def load_sharegpt(self, path: str) -> List[Dict[str, Any]]:
        data = self.load_json(path)
        conversations = []
        for item in data:
            conv = []
            for turn in item.get("conversations", []):
                role = "user" if turn.get("from") in ("human", "user") else "bot"
                conv.append({"role": role, "content": turn.get("value", "")})
            conversations.append({"conversation": conv, **{k: v for k, v in item.items() if k != "conversations"}})
        return conversations

    def load_alpaca(self, path: str) -> List[Dict[str, Any]]:
        data = self.load_json(path)
        conversations = []
        for item in data:
            conv = []
            if item.get("instruction"):
                conv.append({"role": "user", "content": item["instruction"]})
            if item.get("input"):
                conv[-1]["content"] += "\n" + item["input"]
            if item.get("output"):
                conv.append({"role": "bot", "content": item["output"]})
            conversations.append({"conversation": conv})
        return conversations

    def load_messages(self, path: str) -> List[Dict[str, Any]]:
        data = self.load_json(path)
        conversations = []
        for item in data:
            messages = item.get("messages", [])
            if not messages:
                continue
            conv = []
            for msg in messages:
                role = msg.get("role", "user")
                mapped_role = "bot" if role == "assistant" else role
                conv.append({"role": mapped_role, "content": msg.get("content", "")})
            conversations.append({"conversation": conv})
        return conversations

    def load_csv(self, path: str) -> List[Dict[str, Any]]:
        import csv
        conversations = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conv = []
                if row.get("context") and row.get("response"):
                    conv.append({"role": "user", "content": row["context"]})
                    conv.append({"role": "bot", "content": row["response"]})
                conversations.append({"conversation": conv})
        return conversations

    def load_parquet(self, path: str) -> List[Dict[str, Any]]:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        df = table.to_pandas()
        conversations = []
        for _, row in df.iterrows():
            conv = []
            if row.get("context") and row.get("response"):
                conv.append({"role": "user", "content": str(row["context"])})
                conv.append({"role": "bot", "content": str(row["response"])})
            conversations.append({"conversation": conv})
        return conversations

    def format_conversation(self, conversation: List[Dict[str, str]], add_bos: bool = True) -> str:
        parts = []
        if add_bos:
            parts.append(self.SPECIAL_TOKENS["bos"])
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "system":
                parts.append(f"{self.SPECIAL_TOKENS['system']} {content}")
            elif role == "user":
                parts.append(f"{self.SPECIAL_TOKENS['user']} {content}")
            elif role == "bot":
                parts.append(f"{self.SPECIAL_TOKENS['bot']} {content}")
            elif role == "tool":
                parts.append(f"{self.SPECIAL_TOKENS['tool']} {content}")
        return " ".join(parts)

    def build_multi_turn_examples(
        self,
        conversations: List[Dict[str, Any]],
        sliding_window: int = 4,
    ) -> List[Dict[str, Any]]:
        examples = []
        for item in conversations:
            if "context" in item and "response" in item:
                examples.append(item)
                continue
            conv = item.get("conversation", [])
            if len(conv) < 2:
                continue
            for i in range(1, len(conv)):
                context_start = max(0, i - sliding_window)
                context_turns = conv[context_start:i]
                target_turn = conv[i]
                if target_turn.get("role") not in ("bot", "assistant"):
                    continue
                examples.append({
                    "context": self.format_conversation(context_turns, add_bos=True),
                    "response": target_turn.get("content", ""),
                    "metadata": {k: v for k, v in item.items() if k != "conversation"},
                })
        return examples

    def _peek_format(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                sample = json.load(f)
            if isinstance(sample, list) and len(sample) > 0:
                if "messages" in sample[0]:
                    return "messages"
                if "conversations" in sample[0]:
                    return "sharegpt"
                if "instruction" in sample[0]:
                    return "alpaca"
            return "json"
        except Exception:
            return "json"

    def load_and_build(
        self,
        path: str,
        format: str = "auto",
        sliding_window: int = 4,
    ) -> List[Dict[str, Any]]:
        path_str = str(path)
        if format == "auto":
            if path_str.endswith(".jsonl"):
                format = "jsonl"
            elif path_str.endswith(".json"):
                format = "json"
            elif path_str.endswith(".txt"):
                format = "txt"
            elif path_str.endswith(".csv"):
                format = "csv"
            elif path_str.endswith(".parquet"):
                format = "parquet"

        if format == "json":
            sample = self._peek_format(path_str)
            if sample == "messages":
                format = "messages"

        loaders = {
            "json": self.load_json,
            "jsonl": self.load_jsonl,
            "txt": self.load_txt,
            "sharegpt": self.load_sharegpt,
            "alpaca": self.load_alpaca,
            "messages": self.load_messages,
            "csv": self.load_csv,
            "parquet": self.load_parquet,
        }

        if format in loaders:
            data = loaders[format](path)
        else:
            data = self.load_json(path)

        user_assistant_mapped = []
        for item in data:
            if isinstance(item, dict):
                if "context" in item and "response" in item:
                    user_assistant_mapped.append(item)
                    continue
                conv = item.get("conversation", item)
                if isinstance(conv, list) and all(isinstance(t, dict) and "role" in t for t in conv):
                    user_assistant_mapped.append(item)
                elif isinstance(conv, list) and all(isinstance(t, str) for t in conv):
                    new_conv = []
                    for j, t in enumerate(conv):
                        role = "user" if j % 2 == 0 else "bot"
                        new_conv.append({"role": role, "content": t})
                    user_assistant_mapped.append({"conversation": new_conv})
                else:
                    user_assistant_mapped.append(item)
            elif isinstance(item, list):
                new_conv = []
                for j, t in enumerate(item):
                    role = "user" if j % 2 == 0 else "bot"
                    new_conv.append({"role": role, "content": str(t)})
                user_assistant_mapped.append({"conversation": new_conv})

        return self.build_multi_turn_examples(user_assistant_mapped, sliding_window=sliding_window)
