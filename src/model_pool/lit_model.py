import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tape.utils import setup_optimizer
from torch import nn

from src.model_pool.model import MTLTransformerEncoder
from src.util import SequentialLR


class LitMTL(pl.LightningModule):
    def __init__(self, mtl_config, bert_config):
        super().__init__()
        self.mtl_config = mtl_config

        self.model = MTLTransformerEncoder(
            bert_config, mtl_config.mode, mtl_config.tasks
        )

    def predict_step(self, batch, batch_idx):
        t = 'iRT'
        (t_out,) = self.model(batch["token_ids"], task=t)
        return t_out


    def configure_optimizers(self):
        if self.mtl_config.optim == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=float(self.mtl_config.lr),
                momentum=0.9,
                nesterov=True,
            )
        elif self.mtl_config.optim == "adamw":
            # Uses TAPE AdamW optimizer, not all parameters have weight decay
            optimizer = setup_optimizer(self.model, self.mtl_config.lr)
        elif self.mtl_config.optim == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=float(self.mtl_config.lr)
            )
        else:
            raise NotImplementedError(f"Optimizer {self.mtl_config.optim}")

        def calc_decay(e):
            if e <= max_e:
                return 1 - (e * (1 - min_fact) / max_e)
            else:
                return min_fact

        if self.mtl_config.scheduler == "warmup_decay_cos":
            max_e = self.mtl_config.decay_epochs
            min_fact = 0.1
            cosine_scheduler_wr = (
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    self.mtl_config.cos_freq_epochs,
                    eta_min=self.mtl_config.lr * 0.3,
                )
            )
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.mtl_config.warmup_epochs,
                verbose=True,
            )
            lin_decay = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer, calc_decay, verbose=True
            )
            decay_cos = torch.optim.lr_scheduler.ChainedScheduler(
                [cosine_scheduler_wr, lin_decay]
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warm_up, decay_cos],
                milestones=[self.mtl_config.warmup_epochs],
                verbose=True,
            )
            return [optimizer], [scheduler]
        elif self.mtl_config.scheduler == "warmup":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.mtl_config.warmup_epochs,
                verbose=True,
            )
            return [optimizer], [scheduler]
        elif self.mtl_config.scheduler == "none":
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {self.mtl_config.scheduler}")

