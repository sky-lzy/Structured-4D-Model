from __future__ import annotations

import copy
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from policy.common.json_logger import JsonLogger
from policy.common.pytorch_util import optimizer_to
from policy.dataset.hdf5_dataset import HDF5Dataset
from policy.model.common.lr_scheduler import get_scheduler
from policy.model.diffusion.ema_model import EMAModel
from policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")

    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        super().__init__(cfg, output_dir=output_dir)

        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetImagePolicy | None = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.checkpoint_every = 1

        dataset = HDF5Dataset(output_dir=self.output_dir, **cfg.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        if len(train_dataloader) == 0:
            raise RuntimeError("Training dataloader is empty; check dataset path and batch size.")

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max(
                1,
                (len(train_dataloader) * cfg.training.num_epochs)
                // cfg.training.gradient_accumulate_every,
            ),
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel | None = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        log_path = Path(self.output_dir) / "logs.json.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with JsonLogger(str(log_path)) as json_logger:
            epoch_bar = tqdm.tqdm(total=cfg.training.num_epochs, desc=f"Training epoch {self.epoch}")
            for _ in range(cfg.training.num_epochs):
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)
                else:
                    self.model.train()

                train_losses: list[float] = []
                self.optimizer.zero_grad(set_to_none=True)

                for batch_idx, batch in enumerate(train_dataloader):
                    obs = {key: value.to(device) for key, value in batch["obs"].items()}
                    actions = batch["action"].to(device)
                    raw_loss = self.model.compute_loss({"obs": obs, "action": actions})
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    is_accumulation_step = (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0
                    is_last_batch = batch_idx == len(train_dataloader) - 1
                    if is_accumulation_step or is_last_batch:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        lr_scheduler.step()
                        if ema is not None:
                            ema.step(self.model)

                    train_loss = float(raw_loss.item())
                    train_losses.append(train_loss)
                    step_log = {
                        "train_loss": train_loss,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    json_logger.log(step_log)
                    epoch_bar.set_postfix(loss=train_loss, refresh=False)
                    self.global_step += 1

                    if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps - 1:
                        break

                if not train_losses:
                    raise RuntimeError("No training batches were produced.")

                epoch_log = {
                    "train_loss": float(np.mean(train_losses)),
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                json_logger.log(epoch_log)

                if (self.epoch + 1) % cfg.training.checkpoint_every == 0:
                    checkpoint_dir = Path(self.output_dir) / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    self.save_checkpoint(path=checkpoint_dir / f"{self.epoch + 1}.ckpt")

                self.epoch += 1
                epoch_bar.update(1)

        if self._saving_thread is not None:
            self._saving_thread.join()
