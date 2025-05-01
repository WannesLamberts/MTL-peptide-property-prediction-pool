import pandas as pd
import os
from sklearn.metrics import median_absolute_error,median_absolute_error

def create_dataset(file, out_file):
    df = pd.read_parquet(file, engine="pyarrow")
    # Select relevant columns
    df = df[['sequence', 'iRT', 'filename','dataset']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label', 'filename', 'dataset']

    # Add missing columns
    df['task'] = 'iRT'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    df = df.reset_index(drop=True)
    df.to_parquet(out_file, index=True)

def create_dataset_df(df, out_file):
    # Select relevant columns
    df = df[['sequence', 'iRT', 'filename','dataset','task_id']].copy()

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label', 'filename', 'dataset','task_id']

    # Add missing columns
    df['task'] = 'iRT'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    df = df.reset_index(drop=True)
    df.to_parquet(out_file, index=True)


def create_dataset_encoding(df):

    # Select relevant columns
    df = df[['sequence', 'iRT']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label']

    df['Charge'] = ''  # Empty values
    df['DCCS_sequence'] = ''  # Empty values
    return df

def get_loss_predictions_file(file):
    df = pd.read_csv(file, index_col=0)
    mae = median_absolute_error(df['label'], df['predictions'])
    return mae

def get_loss_predictions2(dir):
    losses ={}
    for file in os.listdir(dir):
        filepath = os.path.join(dir, file)  # Full file path
        losses[file] = get_loss_predictions_file(filepath)
    return losses

import numpy as np
def cross_validate(all_data, n_splits=5, output_dir="", random_state=42):
    df = pd.read_parquet(all_data, engine="pyarrow")
    # Params
    n_val_groups = 1
    n_test_groups = 1
    random_state = 42

    # Shuffle groups
    groups = df["sequence"].unique()
    groups = np.random.RandomState(seed=random_state).permutation(groups)

    # Total number of groups per fold
    groups_per_fold = n_val_groups + n_test_groups
    n_folds = len(groups) // groups_per_fold

    # Cross-validation loop
    for fold in range(n_folds):
        fold_start = fold * groups_per_fold
        fold_end = fold_start + groups_per_fold

        val_groups = groups[fold_start: fold_start + n_val_groups]
        test_groups = groups[fold_start + n_val_groups: fold_end]
        train_groups = np.setdiff1d(groups, np.concatenate([val_groups, test_groups]))

        train_idx = df[df["sequence"].isin(train_groups)].index
        val_idx = df[df["sequence"].isin(val_groups)].index
        test_idx = df[df["sequence"].isin(test_groups)].index

        print(f"Fold {fold + 1}:")
        print(f"  Train groups: {train_groups}")
        print(f"  Val groups: {val_groups}")
        print(f"  Test groups: {test_groups}")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")

if __name__ == "__main__":
    cross_validate("../raw_data/80_tasks.parquet")