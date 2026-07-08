import argparse
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_gpt.config import ModelConfig
from simple_gpt.models.model import SimpleGPT
from simple_gpt.inference.chat import ChatSession


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained GPT model")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_latest.pt")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--system-prompt", default=None, help="System prompt for chat")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer file")
    parser.add_argument("--memory-window", type=int, default=10)
    parser.add_argument("--stream", action="store_true", help="Stream output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.checkpoint):
        alt_path = "checkpoint.pt"
        if os.path.exists(alt_path):
            args.checkpoint = alt_path
        else:
            print(f"Checkpoint not found at {args.checkpoint}")
            print("Train a model first: python train.py --data data/pairs.json")
            return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_config") or ModelConfig(**ckpt["config"])
    if isinstance(model_cfg, dict):
        model_cfg = ModelConfig(**model_cfg)

    model = SimpleGPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint with config: n_layer={model_cfg.n_layer}, n_head={model_cfg.n_head}, n_embd={model_cfg.n_embd}")

    from simple_gpt.tokenizer.word import WordTokenizer
    from simple_gpt.tokenizer.bpe import BPETokenizer
    tok_vocab = ckpt.get("tokenizer_vocab") or ckpt.get("vocab")
    if tok_vocab:
        has_bpe_merges = any(isinstance(k, str) and "," in k for k in (ckpt.get("merges") or {}).keys())
        tok_cls = BPETokenizer if has_bpe_merges else WordTokenizer
        tok = tok_cls()
        tok._load_vocab(tok_vocab)
        if hasattr(tok, "merges") and "merges" in ckpt:
            tok.merges = {tuple(k.split(",")): v for k, v in ckpt["merges"].items()}
        tok.special_ids = ckpt.get("tokenizer_special_ids", {})
        print(f"Restored tokenizer from checkpoint, vocab size: {len(tok)}")
    elif args.tokenizer and os.path.exists(args.tokenizer):
        tok = WordTokenizer.load(args.tokenizer)
        print(f"Loaded tokenizer from {args.tokenizer}, vocab size: {len(tok)}")
    else:
        print("No tokenizer found in checkpoint or path.")
        return

    session = ChatSession(
        model=model,
        tokenizer=tok,
        device=device,
        system_prompt=args.system_prompt,
        memory_window=args.memory_window,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stream=args.stream,
    )

    print("\nChatbot ready. Type 'quit' to exit, 'clear' to reset history.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "clear":
            session.clear_history()
            print("History cleared.")
            continue
        if not text:
            continue

        response = session.chat(text)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
