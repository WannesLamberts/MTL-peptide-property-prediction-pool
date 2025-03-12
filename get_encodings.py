import os
import pickle
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tape.models.modeling_bert import ProteinBertConfig
from torch.utils.data import DataLoader

from src.dataset import MTLPepDataset, custom_collate
from src.lit_model import LitMTL

from src.util import (
    DEFAULT_CONFIG,
    check_checkpoint_path,
    split_run_config,
)

def get_encoding(run, args, run_config):
    args.vocab = pickle.load(open(args.vocab_file, "rb"))
    args.scalers = pickle.load(open(args.scalers_file, "rb"))

    args.df_test = args.all_data

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
        strict = False
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
    predictions = trainer.predict(lit_model, dataloaders=predict_dl)
    return predictions

def get_encoding_run(run,all_data):
    data_config = {
        "all_data": all_data,
        "vocab_file": os.path.join(run, "vocab.p"),
        "scalers_file": os.path.join(run, "scalers.p"),
    }
    run_config = split_run_config(run)
    config_dict = DEFAULT_CONFIG | data_config
    args = Namespace(**config_dict)
    args.predict_file_name = "predict"
    args.mode = "pool"
    return get_encoding(run, args, run_config)



import pandas as pd
import torch


def create_model_dataset(df):

    # Select relevant columns
    df = df[['sequence', 'iRT']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label']

    # Add missing columns
    df['task'] = 'iRT'
    df['Charge'] = ''  # Empty values
    df['DCCS_sequence'] = ''  # Empty values
    return df

def get_average_pool(group):
    name = group['filename'].iloc[0]
    df = create_model_dataset(group)
    #df.to_csv(name, index=True)
    best_run = (
        "lightning_logs/CONFIG=mtl_5foldcv_finetune_own_0,TASKS=CCS_iRT,MODE=supervised,PRETRAIN=own,LR=0.0003262821190296,BS=1024,"
        "OPTIM=adamw,LOSS=mae,CLIP=True,ACTIVATION=gelu,SCHED=warmup_decay_cos,SIZE=180,NUMLAYERS=9/version_0"
    )
    pred = get_encoding_run(best_run, df)
    pred = torch.cat(pred, dim=0)  # Assuming batch dimension is 0
    column_averages = pred.mean(dim=0)
    return pd.Series([column_averages.tolist()])

if __name__ == "__main__":
    filename = "og_data/test_data_10_filenames.tsv"
    #filename = "data/sample_1k/all_data.csv"
    df = pd.read_csv(filename, sep="\t")
    lookup_dic = df.groupby('filename').apply(get_average_pool).reset_index()
    lookup_dic.columns=['filename','features']
    lookup_dic.to_parquet('test.parquet',engine='pyarrow')


