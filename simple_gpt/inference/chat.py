from typing import List, Dict, Optional, Iterator
from simple_gpt.inference.generate import Generator


class ChatSession:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        system_prompt: Optional[str] = None,
        memory_window: int = 10,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = 1.1,
        stream: bool = False,
    ):
        self.generator = Generator(model, tokenizer, device)
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.memory_window = memory_window
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.stream = stream
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        self.history = []
        self.generator.reset_kv_cache()

    def build_prompt_ids(self) -> List[int]:
        ids = [self.tokenizer.bos_id()]
        if self.system_prompt:
            ids.append(self.tokenizer.system_id())
            ids.extend(self.tokenizer.encode(self.system_prompt))
        window = self.history[-self.memory_window * 2:] if self.memory_window else self.history
        for msg in window:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                ids.append(self.tokenizer.user_id())
                ids.extend(self.tokenizer.encode(content))
            elif role in ("bot", "assistant"):
                ids.append(self.tokenizer.bot_id())
                ids.extend(self.tokenizer.encode(content))
        ids.append(self.tokenizer.bot_id())
        return ids

    def chat(self, user_input: str) -> str:
        self.add_message("user", user_input)
        input_ids = self.build_prompt_ids()

        result = self.generator.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.tokenizer.eos_id(),
            stream=self.stream,
        )

        if self.stream:
            return self._stream_response(result)
        self.add_message("bot", result)
        return result

    def _stream_response(self, generator: Iterator[str]) -> str:
        full_response = ""
        for chunk in generator:
            full_response = chunk
        self.add_message("bot", full_response)
        return full_response

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def export_history(self) -> str:
        lines = []
        for msg in self.history:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
