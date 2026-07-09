import argparse
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_gpt.config import Config
from simple_gpt.utils.config import load_config
from simple_gpt.utils.seed import set_seed
from simple_gpt.utils.logger import setup_logger
from simple_gpt.tokenizer.word import WordTokenizer
from simple_gpt.tokenizer.bpe import BPETokenizer
from simple_gpt.models.model import SimpleGPT
from simple_gpt.datasets.dataset import ConversationDataset
from simple_gpt.datasets.conversation_builder import ConversationBuilder
from simple_gpt.datasets.preprocess import Preprocessor
from simple_gpt.trainer.trainer import Trainer

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a GPT model from scratch")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--model-config", help="Path to model config YAML (overrides)")
    parser.add_argument("--train-config", help="Path to train config YAML (overrides)")
    parser.add_argument("--data", help="Path to training data")
    parser.add_argument("--tokenizer", help="Path to saved tokenizer (load or save)")
    parser.add_argument("--out", default="checkpoint.pt", help="Output checkpoint path")
    parser.add_argument("--resume", help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--block-size", type=int, help="Max sequence length")
    parser.add_argument("--n-layer", type=int, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, help="Embedding dimension")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--max-steps", type=int, help="Max training steps")
    parser.add_argument("--preprocess-only", action="store_true", help="Only preprocess data, don't train")
    args = parser.parse_args()

    cfg = load_config(
        config_path=args.config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        overrides={
            k: v for k, v in {
                "data.train_path": args.data,
                "train.checkpoint.resume_from": args.resume,
                "train.epochs": args.epochs,
                "train.batch_size": args.batch_size,
                "train.optimizer.lr": args.lr,
                "model.block_size": args.block_size,
                "model.n_layer": args.n_layer,
                "model.n_head": args.n_head,
                "model.n_embd": args.n_embd,
                "train.compile": args.compile,
                "train.seed": args.seed,
                "train.max_steps": args.max_steps,
            }.items() if v is not None
        },
    )
    cfg.train.checkpoint.save_dir = os.path.dirname(args.out) or "checkpoints"

    set_seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    data_path = cfg.data.train_path
    if not os.path.exists(data_path):
        logger.error(f"Training data not found: {data_path}")
        logger.info("Run prepare_data.py first to create training data")
        sys.exit(1)

    builder = ConversationBuilder(seq_len=cfg.data.max_seq_length)
    raw_data = builder.load_and_build(data_path)
    logger.info(f"Loaded {len(raw_data)} examples from {data_path}")

    tokenizer_path = args.tokenizer or cfg.data.tokenizer_path
    tok_cls = BPETokenizer if (args.tokenizer and "bpe" in args.tokenizer) else WordTokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        tok = tok_cls.load(tokenizer_path)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
    else:
        tok = tok_cls()
        texts = []
        for item in raw_data:
            if item.get("context"):
                texts.append(item["context"])
            if item.get("response"):
                texts.append(item["response"])
        tok._build_vocab(texts, min_freq=1, max_vocab_size=cfg.model.vocab_size)
        logger.info(f"Built tokenizer with vocab size {len(tok)}")
        if tokenizer_path:
            os.makedirs(os.path.dirname(tokenizer_path) or ".", exist_ok=True)
            tok.save(tokenizer_path)
            logger.info(f"Saved tokenizer to {tokenizer_path}")

    cfg.model.vocab_size = len(tok)

    if args.preprocess_only:
        logger.info("Preprocessing complete. Exiting (--preprocess-only)")
        return

    train_data = raw_data
    val_data = []

    if cfg.data.val_path and os.path.exists(cfg.data.val_path):
        val_data = builder.load_and_build(cfg.data.val_path)
        logger.info(f"Loaded {len(val_data)} validation examples")

    if not val_data and cfg.train.val_split > 0:
        split_idx = int(len(train_data) * (1 - cfg.train.val_split))
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} val")

    seq_len = cfg.model.block_size
    train_dataset = ConversationDataset(
        train_data, tok,
        seq_len=seq_len,
    )
    val_dataset = None
    if val_data:
        val_dataset = ConversationDataset(
            val_data, tok,
            seq_len=seq_len,
        )

    model = SimpleGPT(cfg.model).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, cfg, tokenizer=tok, device=device)
    trainer.train(train_dataset, val_dataset)

    logger.info(f"Training complete. Best val loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
