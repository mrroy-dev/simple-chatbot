import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 4
    n_embd: int = 384
    dropout: float = 0.1
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    embd_dropout: float = 0.1
    bias: bool = False
    rope_base: int = 10000
    rope_scaling: Optional[dict] = None
    gradient_checkpointing: bool = False
    weight_tying: bool = True
    init_std: float = 0.02
    norm_eps: float = 1e-5
    mlp_ratio: int = 4
    activation: str = "swiglu"


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    fused: bool = True


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 1e-5
    decay_type: str = "cosine"


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_every_steps: int = 1000
    save_every_epochs: int = 1
    keep_last_n: int = 3
    save_best: bool = True
    best_metric: str = "val_loss"
    resume_from: Optional[str] = None


@dataclass
class TrainConfig:
    batch_size: int = 32
    micro_batch_size: Optional[int] = None
    epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    mixed_precision: str = "bf16"
    compile: bool = False
    log_every: int = 10
    val_every: int = 100
    val_split: float = 0.05
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42
    early_stop_patience: Optional[int] = None
    ema_decay: Optional[float] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class DataConfig:
    train_path: str = "data/pairs.json"
    val_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    max_seq_length: int = 512
    dataset_format: str = "pairs"
    streaming: bool = False
    cache_dir: Optional[str] = None


@dataclass
class WandbConfig:
    project: str = "simple-gpt"
    run_name: Optional[str] = None
    entity: Optional[str] = None
    log_every: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: Optional[WandbConfig] = None

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.train, dict):
            train_dict = self.train
            if isinstance(train_dict.get("optimizer"), dict):
                train_dict["optimizer"] = OptimizerConfig(**train_dict["optimizer"])
            if isinstance(train_dict.get("scheduler"), dict):
                train_dict["scheduler"] = SchedulerConfig(**train_dict["scheduler"])
            if isinstance(train_dict.get("checkpoint"), dict):
                train_dict["checkpoint"] = CheckpointConfig(**train_dict["checkpoint"])
            self.train = TrainConfig(**train_dict)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.wandb, dict):
            self.wandb = WandbConfig(**self.wandb)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def update(self, updates: dict) -> "Config":
        d = self.to_dict()
        for k, v in updates.items():
            keys = k.split(".")
            target = d
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = v
        return Config(**d)
