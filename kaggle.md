# Kaggle Training Guide

Use Kaggle's free GPU (P100/T4/A100) to train Simple GPT.

## Setup

### 1. Create a Notebook

1. Go to [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**
2. Set **Accelerator** → **GPU P100** (or T4 x2 / A100 if available)
3. Set **Persistence** → **Files only** (or Files + outputs if saving checkpoints)

### 2. Upload Dataset

**Option A — Upload as Kaggle Dataset (recommended)**
1. Go to **Datasets** → **New Dataset**
2. Upload `data/conversation.json` (or your training JSON)
3. In the notebook, click **Add Data** → search your dataset → **Add**

**Option B — Direct upload**
1. Click **+ Add Data** → **Upload** → select your JSON file

### 3. Install Dependencies

In the first cell:

```python
!pip install torch>=2.0.0 pyyaml>=6.0

# Clone the repo
!git clone https://github.com/YOUR_USERNAME/simple-chatbot.git
%cd simple-chatbot
```

If you uploaded your data as a Kaggle dataset, copy it in:

```python
!cp /kaggle/input/YOUR-DATASET-NAME/conversation.json data/
```

### 4. Prepare Data (optional, if using raw messages format)

```python
!python prepare_data.py --input data/conversation.json --output data/pairs.json
```

## Training

### Quick Start (small model)

```python
!python train.py \
    --data data/pairs_train.json \
    --epochs 10 \
    --batch-size 16 \
    --n-layer 6 \
    --n-head 8 \
    --n-embd 384 \
    --block-size 256 \
    --compile
```

### Full config

```python
!python train.py \
    --config configs/default.yaml \
    --data data/pairs_train.json \
    --val-data data/pairs_val.json \
    --epochs 30 \
    --batch-size 32 \
    --lr 3e-4 \
    --compile
```

### With tokenizer

```python
!python train.py \
    --data data/pairs_train.json \
    --tokenizer data/tokenizer/tokenizer.json \
    --epochs 20 \
    --compile
```

### Resume from checkpoint

```python
!python train.py \
    --data data/pairs_train.json \
    --resume checkpoints/checkpoint_latest.pt \
    --epochs 50 \
    --compile
```

## Saving & Downloading Outputs

### Checkpoints

Checkpoints are saved to `checkpoints/` by default. After training, zip and save:

```python
!zip -r checkpoints.zip checkpoints/
```

Then click the **Output** tab on the right → tick `checkpoints.zip` → **Download**.

### Model only (for inference)

```python
import torch
from simple_gpt.models.model import SimpleGPT
from simple_gpt.config import ModelConfig

ckpt = torch.load("checkpoints/checkpoint_best.pt")
cfg = ckpt["model_config"]
model = SimpleGPT(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Save just the model
torch.save({
    "model_state": model.state_dict(),
    "model_config": cfg,
}, "simple-gpt-model.pt")
```

## Kaggle Specs

| Resource | Free Tier |
|----------|-----------|
| GPU | NVIDIA Tesla P100 16GB |
| CPU | 4 cores |
| RAM | 29 GB |
| Disk | 73 GB (ephemeral) |
| Session | 9 hours (GPU) / 12 hours (CPU) |
| Internet | Enabled |

## Tips

- **Use `--compile`** for ~30-40% speedup (PyTorch 2.0+)
- **Reduce `--batch-size`** if you hit GPU OOM (start with 8-16)
- **Use `--block-size 256`** instead of 512 to save memory
- **Save checkpoints** frequently — Kaggle sessions may time out
- **Kaggle Datasets** → create a dataset version with your trained checkpoint for easy reuse
- Use `nvidia-smi` to monitor GPU usage: `!nvidia-smi`
