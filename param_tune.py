import pandas as pd
from argparse import Namespace

from src.util import (
    DEFAULT_CONFIG,
)
from train import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning configuration or create new configurations"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-i", type=int, default=None)
    parser.add_argument(
        "-p",
        "--pretrained-model",
        default="none",
        type=str,
        choices=["none", "tape", "own"],
    )
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--train-i", type=str)
    parser.add_argument("--val-i", type=str)
    parser.add_argument("--test-i", type=str)
    parser.add_argument("--vocab-file", type=str)
    parser.add_argument("--scalers-file", type=str)
    parser.add_argument(
        "--checkpoint-id",
        type=int,
        default=None,
        help="Index of the checkpoint path to use from config/checkpoints.csv. This is an alternative to "
        "giving the full checkpoint path with '--checkpoint-path'",
    )
    parser.add_argument(
        "--lookup",
        default=None,
        type=str,
        help="the lookup table for the pools",
    )
    args = parser.parse_args()

    # if args.hpt_mode == "train" and args.i is None:
    #     raise ValueError("Argument -i is required when --hpt-mode is train")

    args.hpt_config = None

    return args


def train_from_config(**kwargs):
    config = kwargs["config"]
    i = kwargs["i"]

    config_dict = (
        pd.read_csv(f"hpt/{config}.csv", index_col=False).iloc[i].to_dict()
    )
    config_dict = DEFAULT_CONFIG | kwargs | config_dict | {"config": config}
    run_config = Namespace(**config_dict)
    run_config = post_process_args(run_config)
    run_config.name += f",TRIALID={i}"
    train(run_config)

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.expand_frame_repr", False)  # Don't wrap columns
    pd.set_option("display.max_colwidth", 100)
    args = parse_args()

    train_from_config(**args.__dict__)

