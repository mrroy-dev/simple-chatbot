import os
import torch
import glob
from typing import Optional, Any
from pathlib import Path
from simple_gpt.config import CheckpointConfig


class CheckpointManager:
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, tag: str) -> str:
        return str(self.save_dir / f"checkpoint_{tag}.pt")

    def _best_path(self) -> str:
        return str(self.save_dir / "checkpoint_best.pt")

    def _latest_path(self) -> str:
        return str(self.save_dir / "checkpoint_latest.pt")

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        loss: float,
        config: Config,
        tokenizer,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        tag: Optional[str] = None,
    ):
        import subprocess
        try:
            git_version = subprocess.check_output(
                ["git", "describe", "--always", "--dirty"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_version = "unknown"

        state = {
            "model_state": model.state_dict() if hasattr(model, "state_dict") else model,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "config": config.to_dict(),
            "model_config": config.model,
            "train_config": config.train,
            "tokenizer": tokenizer,
            "git_version": git_version,
            "metrics": metrics or {},
        }

        path = self._checkpoint_path(tag) if tag else self._latest_path()
        torch.save(state, path)
        if tag:
            torch.save(state, self._latest_path())

        if is_best:
            torch.save(state, self._best_path())

        self._cleanup_old()

    def load_latest(self, device: str = "cpu") -> Optional[Dict[str, Any]]:
        path = self._latest_path()
        if os.path.exists(path):
            return torch.load(path, map_location=device, weights_only=False)
        return None

    def load_best(self, device: str = "cpu") -> Optional[Dict[str, Any]]:
        path = self._best_path()
        if os.path.exists(path):
            return torch.load(path, map_location=device, weights_only=False)
        return None

    def load(self, path: str, device: str = "cpu") -> Optional[Dict[str, Any]]:
        if os.path.exists(path):
            return torch.load(path, map_location=device, weights_only=False)
        return None

    def resume_from(self, path: str, device: str = "cpu") -> Optional[Dict[str, Any]]:
        return self.load(path, device)

    def _cleanup_old(self):
        if self.config.keep_last_n <= 0:
            return
        pattern = str(self.save_dir / "checkpoint_epoch_*.pt")
        checkpoints = sorted(glob.glob(pattern), key=os.path.getctime)
        while len(checkpoints) > self.config.keep_last_n:
            os.remove(checkpoints.pop(0))

    def list_checkpoints(self) -> list:
        return sorted(glob.glob(str(self.save_dir / "checkpoint_*.pt")))

    def get_best_metric(self) -> Optional[float]:
        best = self.load_best()
        if best and "metrics" in best:
            metric = self.config.best_metric
            return best["metrics"].get(metric)
        return None
