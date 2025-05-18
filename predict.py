import argparse
import os
import pickle
import sys
from argparse import Namespace

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tape.models.modeling_bert import ProteinBertConfig
from torch.utils.data import DataLoader

from src.dataset import MTLPepDataset, custom_collate
from src.lit_model import LitMTL
from src.read_data import apply_index_file
from src.util import (
    DEFAULT_CONFIG,
    check_checkpoint_path,
    split_run_config,
)


def predict(run, args, run_config):
    args.vocab = pickle.load(open(args.vocab_file, "rb"))
    args.scalers = pickle.load(open(args.scalers_file, "rb"))
    all_data_df = pd.read_parquet(args.all_data_file)
    #args.df_test = all_data_df
    args.df_test = apply_index_file(all_data_df, args.predict_i)

    lookup_df = pd.read_parquet(args.lookup_file, engine='pyarrow')
    args.df_test = args.df_test.merge(lookup_df, on='filename', how='left')
    predict_ds = MTLPepDataset(args.df_test, args)
    predict_dl = DataLoader(
        predict_ds,
        batch_size=args.bs,
        collate_fn=custom_collate,
        num_workers=1,
    )

    bert_config = ProteinBertConfig.from_pretrained(
        "bert-base",
        vocab_size=len(args.vocab),
        hidden_act=run_config["ACTIVATION"],
        hidden_size=int(run_config["SIZE"]),
        intermediate_size=int(run_config["SIZE"]) * 4,
        num_hidden_layers=int(run_config["NUMLAYERS"]),
    )
    hidden_size_mlp =[int(size) for size in run_config["HIDDENSIZEMLP"].split("-")]

    bert_config.dropout_mlp = float(run_config["DROPOUTMLP"])
    bert_config.hidden_size_mlp = hidden_size_mlp
    bert_config.activation_mlp = run_config["ACTIVATIONMLP"]

    lit_model = LitMTL.load_from_checkpoint(
        check_checkpoint_path(os.path.join(run, "checkpoints")),
        mtl_config=args,
        bert_config=bert_config,
    )

    name, version = run.split("/")[-2:]
    logger = TensorBoardLogger("./lightning_logs", name=name, version=version)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        logger=logger,
        strategy=(
            "ddp_find_unused_parameters_true" if args.gpus > 1 else "auto"
        ),
        precision="16-mixed",
    )
    trainer.test(lit_model, dataloaders=predict_dl)


def predict_run(run, all_data_file, predict_i,lookup_file):
    data_config = {
        "all_data_file": all_data_file,
        "predict_i": predict_i,
        "vocab_file": os.path.join(run, "vocab.p"),
        "scalers_file": os.path.join(run, "scalers.p"),
        "lookup_file": lookup_file,
    }

    run_config = split_run_config(run)
    config_dict = DEFAULT_CONFIG | data_config
    args = Namespace(**config_dict)
    args.predict_file_name = "predict"
    args.type = run_config["TYPE"]
    predict(run, args, run_config)


if __name__ == "__main__":
    # Setup argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Run prediction with a trained model")

    # Add arguments to the parser
    parser.add_argument(
        "--run", type=str, required=True, help="Path to the model run (e.g., best_run directory)"
    )
    parser.add_argument(
        "--all_data_file", type=str, required=True, help="Path to the all data CSV file"
    )
    parser.add_argument(
        "--predict_i", type=str, required=True, help="Path to the prediction index file"
    )
    parser.add_argument(
        "--lookup_file", type=str, required=True, help="Path to the prediction index file"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the prediction function with command-line arguments
    predict_run(args.run, args.all_data_file, args.predict_i,args.lookup_file)