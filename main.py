from get_encodings import *
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

def convert(group):
    name = group['filename'].iloc[0]
    df = create_model_dataset(group)
    df.to_csv(name, index=True)
    best_run = (
        "lightning_logs/CONFIG=mtl_5foldcv_finetune_own_0,TASKS=CCS_iRT,MODE=supervised,PRETRAIN=own,LR=0.0003262821190296,BS=1024,"
        "OPTIM=adamw,LOSS=mae,CLIP=True,ACTIVATION=gelu,SCHED=warmup_decay_cos,SIZE=180,NUMLAYERS=9/version_0"
    )
    pred = get_encoding_run(best_run, df)
    pred = torch.cat(pred, dim=0)  # Assuming batch dimension is 0
    column_averages = pred.mean(dim=0)
    print(column_averages.shape)
    return 5

if __name__ == "__main__":
    filename = "og_data/test_data_calibrated_merged_1000.tsv"
    #filename = "data/sample_1k/all_data.csv"

    df = pd.read_csv(filename, sep="\t")
    df["test"] = df.groupby('filename').apply(convert)
    #print(df)
