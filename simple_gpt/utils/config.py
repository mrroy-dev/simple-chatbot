import os
import yaml
from typing import Optional
from simple_gpt.config import Config


def load_config(
    config_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
    train_config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> Config:
    cfg = Config()
    if config_path and os.path.exists(config_path):
        cfg = Config.from_yaml(config_path)
    if model_config_path and os.path.exists(model_config_path):
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f)
        if model_cfg:
            cfg.model = Config.__dataclass_fields__["model"].type(**model_cfg)
    if train_config_path and os.path.exists(train_config_path):
        with open(train_config_path) as f:
            train_cfg = yaml.safe_load(f)
        if train_cfg:
            for k, v in train_cfg.items():
                if hasattr(cfg.train, k):
                    setattr(cfg.train, k, v)
    if overrides:
        cfg = cfg.update(overrides)
    return cfg
