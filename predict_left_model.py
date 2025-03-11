import os
import pickle
from argparse import Namespace

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tape.models.modeling_bert import ProteinBertConfig
from torch.utils.data import DataLoader

from src.model_left.dataset import MTLPepDataset, custom_collate
from src.model_left.lit_model import LitMTL
from src.read_data import apply_index_file
from src.util import (
    DEFAULT_CONFIG,
    check_checkpoint_path,
    split_run_config,
)


def predict(run, args, run_config):
    args.vocab = pickle.load(open(args.vocab_file, "rb"))
    args.scalers = pickle.load(open(args.scalers_file, "rb"))

    all_data_df = pd.read_csv(args.all_data_file, index_col=0)
    #args.df_test = all_data_df
    args.df_test = apply_index_file(all_data_df, args.predict_i)

    lookup_df = pd.read_parquet('test.parquet', engine='pyarrow')
    original_index = args.df_test.index
    args.df_test = args.df_test.merge(lookup_df, on='filename', how='left')
    args.df_test.index = original_index  # Restore original index
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


def predict_run(run, all_data_file, predict_i):
    data_config = {
        "all_data_file": all_data_file,
        "predict_i": predict_i,
        "vocab_file": os.path.join(run, "vocab.p"),
        "scalers_file": os.path.join(run, "scalers.p"),
    }

    run_config = split_run_config(run)
    config_dict = DEFAULT_CONFIG | data_config
    args = Namespace(**config_dict)
    args.predict_file_name = "predict"
    predict(run, args, run_config)

def create_model_dataset(df):

    # Select relevant columns
    df = df[['sequence', 'iRT','filename']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label','filename']

    # Add missing columns
    df['task'] = 'iRT'
    df['Charge'] = ''  # Empty values
    df['DCCS_sequence'] = ''  # Empty values
    return df


if __name__ == "__main__":

    # Example on how to create predictions with an existing model
    best_run = (
        "lightning_logs/CONFIG=mtl_5foldcv_finetune_own_0,TASKS=CCS_iRT,MODE=supervised,PRETRAIN=own,LR=0.0003262821190296,BS=1024,OPTIM=adamw,LOSS=mae,CLIP=True,ACTIVATION=gelu,SCHED=warmup_decay_cos,SIZE=180,NUMLAYERS=9/version_1"
    )

    predict_run(
        best_run,
        "data/sample_10_filenames/all_data.csv",
        "data/sample_10_filenames/test_0.csv",
    )
