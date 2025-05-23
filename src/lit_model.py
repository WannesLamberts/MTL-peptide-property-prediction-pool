import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tape.utils import setup_optimizer
from torch import nn

from src.model import MTLTransformerEncoder
from src.util import SequentialLR


class LitMTL(pl.LightningModule):
    def __init__(self, mtl_config, bert_config):
        super().__init__()
        self.mtl_config = mtl_config

        self.model = MTLTransformerEncoder(
            bert_config, mtl_config.mode,mtl_config.type
        )

        self.val_predictions, self.test_predictions = (None for _ in range(2))

    def on_train_epoch_start(self):
        self.log(
            f"z_lr",
            self.optimizers().optimizer.param_groups[0]["lr"],
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        if self.mtl_config.mode == "supervised":
            self.val_predictions = ([], [], [])

    def on_test_epoch_start(self):
        if self.mtl_config.mode == "supervised":
            self.test_predictions = ([], [], [])

    def training_step(self, batch, batch_idx):
        loss = self.supervised_step(batch, batch_idx, None)
        self.log(
            f"train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        return loss

    def predict_step(self, batch, batch_idx):
        (t_out,) = self.model(batch["token_ids"])
        return t_out,batch["indx"]

    def validation_step(self, batch, batch_idx):
        loss = self.supervised_step(batch, batch_idx, "val_predictions")
        self.log(
            f"val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.supervised_step(batch, batch_idx, "test_predictions")
        self.log(
            f"test_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        return loss

    def on_validation_epoch_end(self):
        self.end_epoch_predictions("val", self.val_predictions)

    def on_test_epoch_end(self):
        self.end_epoch_predictions("test", self.test_predictions)

    @staticmethod
    def pretrain_loss_metrics(labels, out):
        return nn.CrossEntropyLoss(ignore_index=0)(out.transpose(1, 2), labels)

    def supervised_loss(self, labels, out):
        out = out.squeeze(1)
        targets = labels.float()
        if len(targets) == 0 or len(out) == 0:
            warnings.warn(
                f"No samples in this batch, consider increasing the batch size"
            )
            return torch.tensor(0.0)

        if self.mtl_config.loss == "mse":
            loss = nn.MSELoss()(out, targets)
        elif self.mtl_config.loss == "mae":
            loss = nn.L1Loss()(out, targets)
        else:
            raise NotImplementedError(
                f"Unknown loss function {self.mtl_config.loss}"
            )
        return loss

    def supervised_step(self, batch, batch_idx, save_arg):
        if self.mtl_config.type=="base":
            (t_out,) = self.model(batch["token_ids"])
        elif self.mtl_config.type=="pool":
            (t_out,) = self.model(batch["token_ids"], features=batch["features"])

        if len(t_out) > 0:
            # t_out will be optimized to be equal to the standardized label, inverse transform to get original label
            predictions = (
                self.mtl_config.scalers
                .inverse_transform(t_out.cpu().detach())
                .squeeze(1)
            )
        else:
            predictions = []
        if save_arg is not None:
            # Save index, prediction (and at the end, loss)
            idxs, preds, losss = getattr(self, save_arg)
            to_save = (
                idxs
                + [e for i, e in enumerate(batch["indx"])],
                preds + list(predictions),
                losss,
            )
            setattr(self, save_arg, to_save)
        loss = self.supervised_loss(
            batch["standardized_label"], t_out
        )

        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(
                f"Loss became NaN or Inf in epoch {self.current_epoch} batch {batch_idx}"
            )

        if save_arg is not None:
            # Add the loss to the val/test predictions, this will be used to only save the best epoch
            idxs, preds, losss = getattr(self, save_arg)
            to_save = (idxs, preds, losss + [loss.cpu().item()])
            setattr(self, save_arg, to_save)

        return loss

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

    def end_epoch_predictions(self, data_split, predictions):
        if predictions is None:
            return

        if (
            "predict_file_name" not in self.mtl_config
            or self.mtl_config.predict_file_name is None
        ):
            predict_file_name = data_split
        else:
            predict_file_name = self.mtl_config.predict_file_name

        loss = np.mean(predictions[-1])
        result_dir = os.path.join(self.logger.log_dir, "predictions")
        os.makedirs(result_dir, exist_ok=True)
        losses = []
        for f in os.listdir(result_dir):
            if predict_file_name in f:
                losses.append(float(f[:-4].split("=")[1]))

        if len(losses) == 0:
            pickle.dump(
                self.mtl_config.scalers,
                open(os.path.join(self.logger.log_dir, f"scalers.p"), "wb"),
            )
            pickle.dump(
                self.mtl_config.vocab,
                open(os.path.join(self.logger.log_dir, f"vocab.p"), "wb"),
            )

        if not self.trainer.sanity_checking and (
            len(losses) == 0 or loss < min(losses)
        ):
            for f in os.listdir(result_dir):
                if predict_file_name in f:
                    os.remove(os.path.join(result_dir, f))
            idxs, preds, losss = predictions
            predictions_df = pd.DataFrame({"indx": idxs, "predictions": preds})
            data_df = (
                self.mtl_config.df_val
                if predict_file_name == "val"
                else self.mtl_config.df_test
            )
            if self.mtl_config.type=="pool":
                predictions_df = self._add_predictions_to_data(
                    data_df.drop(columns=['features']), predictions_df
                )
            elif self.mtl_config.type=="base":
                predictions_df = self._add_predictions_to_data(
                    data_df, predictions_df
                )
            predictions_df.to_csv(
                os.path.join(
                    result_dir, f"{predict_file_name}_loss={loss:.4f}.csv"
                )
            )

    @staticmethod
    def _add_predictions_to_data(data_df, predictions_df):
        return data_df.join(predictions_df.set_index("indx"), how="left")
