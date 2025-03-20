import os
import pickle
import sys
from argparse import Namespace
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tape.models.modeling_bert import ProteinBertConfig
from torch.utils.data import DataLoader

from src.dataset import MTLPepDataset, custom_collate
from src.lit_model import LitMTL
from src.read_data import apply_index_file
from src.utils_data import create_dataset_encoding
from src.util import (
    DEFAULT_CONFIG,
    check_checkpoint_path,
    split_run_config,
)

def get_encoding(run, args, run_config):
    args.vocab = pickle.load(open(args.vocab_file, "rb"))
    args.scalers = pickle.load(open(args.scalers_file, "rb"))['iRT']

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


def get_mean_pools(df):
    best_run = (
        "lightning_logs/CONFIG=mtl_5foldcv_pretrain_0,TASKS=CCS_iRT,MODE=pretrain,PRETRAIN=none,LR=0.0001940554482365,BS=1024,OPTIM=adam,LOSS=mae,CLIP=False,ACTIVATION=gelu,SCHED=warmup,SIZE=180,NUMLAYERS=9/version_0"
    )
    pred = get_encoding_run(best_run, df)
    # Collect all predictions in a list (preallocate memory if possible)
    batch_list = []
    index_list = []

    for batch_tensor, batch_indices in pred:
        batch_list.append(batch_tensor.numpy())  # Convert batch tensor to NumPy array
        index_list.append(batch_indices)

    # Stack all batches at once
    predictions = np.vstack(batch_list)
    indices = np.concatenate(index_list)

    feature_cols = [f'feature_{i}' for i in range(predictions.shape[1])]
    prediction_df = pd.DataFrame(
        predictions,
        index=indices,
        columns=feature_cols)
    result_df = df.join(prediction_df, how='left')
    mean_by_filename = result_df.groupby('filename')[feature_cols].mean()
    mean_by_filename['features'] = mean_by_filename[feature_cols].values.tolist()
    result = mean_by_filename[['features']]
    return result

if __name__ == "__main__":
    directory = sys.argv[1]
    df = pd.read_csv(directory + "all_data.csv", index_col=0)
    #df = apply_index_file(df, directory + "train_0.csv")
    result = get_mean_pools(df)
    result.to_parquet(directory+'lookup.parquet',engine='pyarrow')








