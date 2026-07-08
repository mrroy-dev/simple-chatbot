import os
import sys
import logging
from typing import Optional


def setup_logger(name: str = "simple_gpt", level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    log_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


class MetricsLogger:
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self.history: dict = {}
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, step: int, metrics: dict):
        for k, v in metrics.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append((step, v))

    def get_latest(self, key: str):
        vals = self.history.get(key, [])
        return vals[-1][1] if vals else None

    def state_dict(self):
        return {"history": self.history}

    def load_state_dict(self, state: dict):
        self.history = state.get("history", {})
