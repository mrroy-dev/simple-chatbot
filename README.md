# Simple GPT

A production-quality GPT training framework built from scratch in PyTorch, inspired by nanoGPT, TinyLlama, SmolLM, and modern open-source LLM projects.

Train a decoder-only transformer on your own conversation data — no pretrained weights, no external APIs.

## Features

### Model
- Pre-LayerNorm (RMSNorm) — like LLaMA
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) / Multi-Query Attention
- SwiGLU activation in FFN
- Weight tying between embeddings and LM head
- Better initialization (residual scaling)
- KV Cache for efficient inference
- Configurable via YAML

### Training
- Mixed precision (bf16/fp16 via AMP)
- Gradient accumulation & clipping
- AdamW with decoupled weight decay
- Warmup + cosine LR schedule
- Checkpoint resume with full state (model, optimizer, scheduler, config, tokenizer)
- Best checkpoint saving
- Early stopping
- EMA (Exponential Moving Average)
- Perplexity, token accuracy, throughput metrics

### Tokenizer
- Byte-Pair Encoding (BPE) built from scratch
- Full special token support: BOS, EOS, PAD, UNK, USER, BOT, SYSTEM, TOOL
- Batch encode/decode with padding & truncation
- Attention masks
- Vocabulary statistics & coverage reports
- Save/load

### Dataset Pipeline
- Multi-turn conversation support via sliding window
- **Official input format: Messages (OpenAI ChatML)** — `{"messages": [{"role": "user"/"assistant", "content": "..."}]}`
- Input formats: JSON, JSONL, TXT, Messages, ShareGPT, Alpaca, CSV
- Text cleaning (URLs, HTML, unicode, emoji, whitespace)
- Data filtering (min/max tokens, empty removal)
- Dataset statistics
- Streaming support for large datasets

### Inference
- Interactive chat with persistent history
- System prompts & memory window
- Streaming generation
- Advanced sampling: top-k, top-p, temperature, repetition penalty
- Stop sequences
- KV cache-aware generation

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Prepare data (messages format is the official input format)
python prepare_data.py --input data/conversation.json --output data/pairs.json

# Train
python train.py --data data/pairs_train.json --epochs 30

# Chat
python chat.py --checkpoint checkpoints/checkpoint_latest.pt
```

## Project Structure

```
simple-chatbot/
├── configs/            # YAML configuration files
│   ├── default.yaml
│   ├── model.yaml
│   └── train.yaml
├── data/               # Data directory
│   ├── raw/
│   ├── processed/
│   └── tokenizer/
├── simple_gpt/         # Main package
│   ├── config.py       # Configuration dataclasses
│   ├── datasets/       # Dataset pipeline
│   │   ├── dataset.py
│   │   ├── cleaners.py
│   │   ├── filters.py
│   │   ├── conversation_builder.py
│   │   └── preprocess.py
│   ├── tokenizer/      # Tokenization
│   │   ├── tokenizer.py
│   │   ├── bpe.py
│   │   └── sentencepiece.py
│   ├── models/         # Transformer model
│   │   ├── model.py
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   ├── mlp.py
│   │   ├── embeddings.py
│   │   └── rotary.py
│   ├── trainer/        # Training system
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── checkpoint.py
│   │   └── evaluator.py
│   ├── inference/      # Inference
│   │   ├── chat.py
│   │   ├── generate.py
│   │   └── sampling.py
│   └── utils/          # Utilities
│       ├── metrics.py
│       ├── logger.py
│       ├── seed.py
│       └── config.py
├── tests/              # Unit tests
├── train.py            # Training entry point
├── chat.py             # Chat entry point
├── prepare_data.py     # Data preparation entry point
├── requirements.txt
└── README.md
```

## Configuration

All hyperparameters are managed via YAML config files:

```bash
# Train with custom config
python train.py --config configs/default.yaml --epochs 50 --batch-size 64

# Use separate model/train configs
python train.py --model-config configs/model.yaml --train-config configs/train.yaml

# Override individual params
python train.py --n-layer 12 --n-head 12 --n-embd 768 --lr 0.0003
```

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/default.yaml` | Main config file |
| `--data` | from config | Training data path |
| `--epochs` | from config | Number of epochs |
| `--batch-size` | from config | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--n-layer` | 6 | Transformer layers |
| `--n-head` | 8 | Attention heads |
| `--n-embd` | 384 | Embedding dimension |
| `--compile` | False | Use torch.compile |
| `--resume` | None | Resume checkpoint path |
| `--seed` | 42 | Random seed |

## Tests

```bash
python -m pytest tests/ -v
```

## Architecture Notes

The model follows the modern LLaMA-style architecture:
- **Pre-normalization** with RMSNorm (more stable than post-norm)
- **Rotary Position Embeddings** (no learned position embeddings)
- **Grouped Query Attention** (fewer KV heads saves memory)
- **SwiGLU** activation (better than ReLU/GELU)
- **Weight tying** (shares embeddings with LM head)
- **No bias** in linear layers (simplifies, saves params)

## License

MIT
