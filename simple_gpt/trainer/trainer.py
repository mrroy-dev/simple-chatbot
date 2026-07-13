import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Any
from simple_gpt.config import Config
from simple_gpt.trainer.optimizer import configure_optimizer
from simple_gpt.trainer.scheduler import get_scheduler
from simple_gpt.trainer.checkpoint import CheckpointManager
from simple_gpt.trainer.evaluator import Evaluator
from simple_gpt.utils.logger import setup_logger, MetricsLogger
from simple_gpt.utils.metrics import throughput

logger = setup_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        tokenizer=None,
        device: str = "cpu",
    ):
        self.model = model
        self.config = config
        self.train_cfg = config.train
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_logger = MetricsLogger()
        self.logger = logger

        self.optimizer = configure_optimizer(model, self.train_cfg)
        self.scheduler = None
        self.checkpoint_mgr = CheckpointManager(self.train_cfg.checkpoint)
        self.evaluator = Evaluator(model, device)
        self.ema_model = None
        self.ema_decay = self.train_cfg.ema_decay

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_step = 0
        self.early_stop_counter = 0

        if self.train_cfg.compile and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(model)
            self.model.to(device)

        self.scaler = None
        mp = self.train_cfg.mixed_precision
        if mp == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            logger.info("Using bfloat16 mixed precision")
        elif mp == "fp16" and torch.cuda.is_available():
            self.amp_dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using float16 mixed precision with gradient scaling")
        else:
            self.amp_dtype = None
            logger.info("Using full precision (fp32)")

        self._resume_checkpoint()

    def _resume_checkpoint(self):
        ckpt_cfg = self.train_cfg.checkpoint
        resume_path = ckpt_cfg.resume_from
        if resume_path:
            ckpt = self.checkpoint_mgr.resume_from(resume_path, self.device)
        else:
            ckpt = self.checkpoint_mgr.load_latest(self.device)

        if ckpt:
            self.model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt and ckpt["optimizer_state"]:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.global_step = ckpt.get("step", 0)
            self.best_val_loss = ckpt.get("metrics", {}).get("val_loss", float("inf"))
            if "metrics_logger" in ckpt:
                self.metrics_logger.load_state_dict(ckpt["metrics_logger"])
            logger.info(f"Resumed from checkpoint: epoch={ckpt.get('epoch', 0)}, step={self.global_step}")

    def _log_dataset_diagnostics(self, train_dataset, val_dataset):
        if self.tokenizer is None:
            return
        contexts = []
        responses = []
        for ds, name in [(train_dataset, "train"), (val_dataset, "val")]:
            if ds is None:
                continue
            for i in range(min(len(ds), 5000)):
                ids, mask = ds.examples[i]
                resp_start = next((j for j, m in enumerate(mask) if m == 1), len(ids))
                ctx = ids[:resp_start]
                resp = [t for t, m in zip(ids[resp_start:], mask[resp_start:]) if m == 1]
                contexts.append(len(ctx))
                responses.append(len(resp))

        avg_ctx = sum(contexts) / max(len(contexts), 1)
        avg_resp = sum(responses) / max(len(responses), 1)
        max_len = max(contexts + responses) if contexts + responses else 0
        min_len = min(contexts + responses) if contexts + responses else 0

        self.logger.info(
            f"Dataset diagnostics | "
            f"Train: {len(train_dataset)} | "
            f"Val: {len(val_dataset) if val_dataset else 0} | "
            f"Avg ctx: {avg_ctx:.1f} tokens | "
            f"Avg resp: {avg_resp:.1f} tokens | "
            f"Max seq: {max_len} | "
            f"Min seq: {min_len}"
        )

    def _build_scheduler(self, total_steps: int):
        self.scheduler = get_scheduler(
            self.optimizer,
            self.train_cfg.scheduler,
            total_steps=total_steps,
        )

    def _get_lr(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        return self.train_cfg.optimizer.lr

    def train_step(self, x, y, m, accum_steps=1):
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                logits, loss, _ = self.model(x, targets=y, loss_mask=m)
        else:
            logits, loss, _ = self.model(x, targets=y, loss_mask=m)

        loss = loss / accum_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        n_batches = len(train_loader)

        accum_steps = self.train_cfg.gradient_accumulation_steps
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            x, y, m, attn = batch
            x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)

            loss = self.train_step(x, y, m, accum_steps)
            total_loss += loss.item() * accum_steps
            total_tokens += m.sum().item()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == n_batches:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.gradient_clip
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

                self.global_step += 1

                if self.ema_decay is not None:
                    self._update_ema()

            if self.global_step % self.train_cfg.log_every == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                tok_per_sec = throughput(total_tokens, elapsed)
                avg_loss = total_loss / max(batch_idx + 1, 1)
                self.logger.info(
                    f"Epoch {self.start_epoch + 1}/{self.train_cfg.epochs} | "
                    f"Batch {batch_idx}/{n_batches} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {self._get_lr():.2e} | "
                    f"tokens/s: {tok_per_sec:.0f}"
                )
                self.metrics_logger.log(self.global_step, {
                    "train_loss": avg_loss,
                    "lr": self._get_lr(),
                    "tokens_per_sec": tok_per_sec,
                })

            if val_loader and self.global_step % self.train_cfg.val_every == 0 and batch_idx > 0:
                self._run_validation(val_loader)

            if self.train_cfg.max_steps and self.global_step >= self.train_cfg.max_steps:
                break

        avg_epoch_loss = total_loss / max(n_batches, 1)
        return avg_epoch_loss

    def _run_validation(self, val_loader: DataLoader):
        metrics = self.evaluator.evaluate(val_loader)
        val_loss = metrics["val_loss"]
        val_ppl = metrics["val_perplexity"]

        self.logger.info(
            f"Validation | "
            f"Loss: {val_loss:.4f} | "
            f"PPL: {val_ppl:.2f} | "
            f"Accuracy: {metrics['val_accuracy']:.4f}"
        )
        self.metrics_logger.log(self.global_step, metrics)

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_step = self.global_step
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        self.checkpoint_mgr.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.start_epoch,
            step=self.global_step,
            loss=val_loss,
            config=self.config,
            tokenizer=self.tokenizer,
            metrics=metrics,
            is_best=is_best,
        )

    def _update_ema(self):
        if self.ema_model is None:
            self.ema_model = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_model[name] = param.data.clone()
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_model:
                    self.ema_model[name] = (
                        self.ema_decay * self.ema_model[name] +
                        (1 - self.ema_decay) * param.data
                    )

    def apply_ema(self):
        if self.ema_model is None:
            return
        for name, param in self.model.named_parameters():
            if name in self.ema_model:
                param.data.copy_(self.ema_model[name])

    def train(
        self,
        train_dataset,
        val_dataset=None,
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            pin_memory=self.train_cfg.pin_memory,
        )
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_cfg.batch_size,
                shuffle=False,
                num_workers=self.train_cfg.num_workers,
                pin_memory=self.train_cfg.pin_memory,
            )

        self._log_dataset_diagnostics(train_dataset, val_dataset)

        total_steps = len(train_loader) * self.train_cfg.epochs
        if self.train_cfg.max_steps:
            total_steps = min(total_steps, self.train_cfg.max_steps)
        self._build_scheduler(total_steps)

        self.logger.info(
            f"Starting training | "
            f"Model params: {sum(p.numel() for p in self.model.parameters()):,} | "
            f"Train examples: {len(train_dataset)} | "
            f"Batch size: {self.train_cfg.batch_size} | "
            f"Total steps: {total_steps}"
        )

        for epoch in range(self.start_epoch, self.train_cfg.epochs):
            epoch_loss = self.train_epoch(train_loader, val_loader)
            self.logger.info(f"Epoch {epoch + 1} completed | Avg loss: {epoch_loss:.4f}")

            ckpt_cfg = self.train_cfg.checkpoint
            if ckpt_cfg.save_every_epochs > 0 and (epoch + 1) % ckpt_cfg.save_every_epochs == 0:
                self.checkpoint_mgr.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    step=self.global_step,
                    loss=epoch_loss,
                    config=self.config,
                    tokenizer=self.tokenizer,
                    tag=f"epoch_{epoch + 1}",
                )

            if self.train_cfg.early_stop_patience and self.early_stop_counter >= self.train_cfg.early_stop_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            if self.train_cfg.max_steps and self.global_step >= self.train_cfg.max_steps:
                self.logger.info(f"Reached max steps {self.train_cfg.max_steps}")
                break

        if self.ema_model:
            self.apply_ema()

        self.checkpoint_mgr.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.train_cfg.epochs - 1,
            step=self.global_step,
            loss=epoch_loss if val_loader else self.best_val_loss,
            config=self.config,
            tokenizer=self.tokenizer,
            tag="final",
        )
        self.logger.info(f"Training complete | Best val loss: {self.best_val_loss:.4f} at step {self.best_step}")
