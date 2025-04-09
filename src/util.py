import argparse
import os
from bisect import bisect_right

import numpy as np
import pandas as pd
from numpy import quantile
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.optim.lr_scheduler import SequentialLR

DEFAULT_CONFIG = {
    "config": None,
    "gpus": 1,
    "mode": "supervised",
    "pretrained_model": "none",
    "checkpoint_path": None,
    "data_file": None,
    "train_i": None,
    "val_i": None,
    "test_i": None,
    "train_file": None,
    "val_file": None,
    "test_file": None,
    "vocab_file": None,
    "scalers_file": None,
    "lr": 0.001,
    "bs": 1024,
    "accumulate_batches": 1,
    "optim": "SGD",
    "loss": "mae",
    "clip_gradients": True,
    "activation": "gelu",
    "hidden_size": 768,
    "num_layers": 12,
    "seq_len": 50,
    "scheduler": "none",
    "warmup_epochs": 10,
    "cos_freq_epochs": 5,
    "decay_epochs": 50,
}


class SequentialLRFix(SequentialLR):
    def __init__(
        self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False
    ):
        super().__init__(
            optimizer, schedulers, milestones, last_epoch, verbose
        )

    def step(self, epoch=None):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step()
        else:
            self._schedulers[idx].step()
        self._last_lr = self._schedulers[idx].get_last_lr()


SequentialLR = SequentialLRFix


def check_checkpoint_path(d):
    if os.path.isdir(d):
        possible_checkpoints = [f for f in os.listdir(d) if f[-5:] == ".ckpt"]
        if len(possible_checkpoints) == 0:
            raise argparse.ArgumentError(
                argument=None,
                message=f"No checkpoint files found in directory {d}",
            )
        val_losses = []
        for f in possible_checkpoints:
            splitted = [e.split("=") for e in f[:-5].split("-")]
            val_loss = [s[1] for s in splitted if s[0] == "val_loss"][0]
            val_losses.append(val_loss)
        best_ckpt = possible_checkpoints[val_losses.index(min(val_losses))]
        return os.path.join(d, best_ckpt)
    elif os.path.isfile(d) and d[-5:] == ".ckpt":
        return d
    else:
        raise argparse.ArgumentError(
            argument=None,
            message=f"Checkpoint path {d} is nor a directory, "
            f"nor a checkpoint file",
        )


def check_data_files(args):
    if args.data_file is not None:
        if any(
            f is not None
            for f in (args.train_file, args.val_file, args.test_file)
        ):
            raise argparse.ArgumentError(
                argument=None,
                message=f"Ambiguous data arguments: both --data-file and "
                f"--train-file, --val-file, or --test-file given",
            )
        if all(f is None for f in (args.train_i, args.val_i, args.test_i)):
            raise argparse.ArgumentError(
                argument=None,
                message=f"Ambiguous data arguments: --data-file given but no "
                f"train, val or test index file",
            )
        return True

    elif all(f is None for f in (args.train_file, args.val_file, args.test_file)):
        raise argparse.ArgumentError(
            argument=None,
            message=f"Ambiguous data arguments: no --data-file, --train-file, "
            f"--val-file, or --test-file given",
        )
    return False


def split_run_config(path):
    dirs = path.split("/")
    for d in dirs:
        if not d.startswith("CONFIG="):
            continue
        settings = d.split(",")
        return {
            setting.split("=")[0]: setting.split("=")[1]
            for setting in settings
        }
    return None


def end_padding(seq, length, pad_token):
    pad_after = length - len(seq)
    if isinstance(seq, str):
        return seq + pad_token * pad_after
    elif isinstance(seq, list):
        return seq + [pad_token] * pad_after
    else:
        raise TypeError(
            f"seq argument should be a string or list, {type(seq)} given"
        )





