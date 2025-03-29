import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
import os
from sklearn.metrics import median_absolute_error,median_absolute_error


def create_dataset_encoding(df):

    # Select relevant columns
    df = df[['sequence', 'iRT']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label']

    df['Charge'] = ''  # Empty values
    df['DCCS_sequence'] = ''  # Empty values
    return df


def create_csv(file):
    df = pd.read_parquet(file, engine="pyarrow")


    # Select relevant columns
    df = df[['sequence', 'iRT', 'filename']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label', 'filename']

    # Add missing columns
    df['task'] = 'iRT'
    return df



def create_dataset(file, out_file, filter_filename=None,amount=None):
    df = pd.read_parquet(file, engine="pyarrow")

    # Select relevant columns
    df = df[['sequence', 'iRT', 'filename']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label', 'filename']

    # Add missing columns
    df['task'] = 'iRT'

    if filter_filename:
        unique_filename_values = df['filename'].unique()[:filter_filename]
        df = df[df['filename'].isin(unique_filename_values)]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if amount:
        df = df.groupby('filename').apply(lambda x: x.sample(frac=amount, random_state=42)).reset_index(drop=True)
    # Reset the index and drop the old index column

    df = df.reset_index(drop=True)
    df.to_csv(out_file, index=True)





def split_data(all_data,train_ratio=0.9,val=0.1,output_dir=""):
    df = pd.read_csv(all_data, index_col=0)
    df.index.to_series().to_csv(output_dir+"encoding_indices.csv", index=False, header=False)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=val, random_state=42)
    train_idx, val_idx = next(gss1.split(
        X=df.drop(columns="label"),
        y=df["label"],
        groups=df["filename"]
    ))
    pd.DataFrame(train_idx, columns=['Index']).to_csv(output_dir+"train.csv", index=False, header=False)
    pd.DataFrame(val_idx, columns=['Index']).to_csv(output_dir+"val.csv", index=False, header=False)

def split_data_val(all_data,train_ratio=0.8, val_ratio=0.1,test_ratio=0.1,output_dir=""):
    df = pd.read_csv(all_data, index_col=0)
    df.index.to_series().to_csv(output_dir+"encoding_indices.csv", index=False, header=False)

    # First split: train vs (val + test)
    test_val_size = val_ratio + test_ratio  # e.g., 0.2 for 80/10/10 split
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_val_size, random_state=42)
    train_idx, test_val_idx = next(gss1.split(
        X=df.drop(columns="label"),
        y=df["label"],
        groups=df["filename"]
    ))

    # Second split: split test_val_idx into val and test
    val_size_relative = val_ratio / (val_ratio + test_ratio)  # e.g., 0.1/0.2 = 0.5
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=42)
    # Use the subset of indices from test_val_idx
    val_idx_relative, test_idx_relative = next(gss2.split(
        X=df.iloc[test_val_idx].drop(columns="label"),
        y=df.iloc[test_val_idx]["label"],
        groups=df.iloc[test_val_idx]["filename"]
    ))

    # Convert relative indices back to original DataFrame indices
    val_idx = test_val_idx[val_idx_relative]
    test_idx = test_val_idx[test_idx_relative]

    # Print or return the indices
    print("Train indices:", train_idx)
    print("Validation indices:", val_idx)
    print("Test indices:", test_idx)

    pd.DataFrame(train_idx, columns=['Index']).to_csv(output_dir+"train.csv", index=False, header=False)
    pd.DataFrame(val_idx, columns=['Index']).to_csv(output_dir+"val.csv", index=False, header=False)
    pd.DataFrame(test_idx, columns=['Index']).to_csv(output_dir+"test.csv", index=False, header=False)
    return train_idx, val_idx, test_idx


def select_train_data(all_data,train,filenames=None):
    all_df = pd.read_csv(all_data,index_col=0)
    train_df = pd.read_csv(train,index_col=0,header=None)

    merged = train_df.merge(all_df,how='left',left_index=True,right_index=True)
    if filenames:
        unique_filename_values = merged['filename'].unique()[:filenames]
        merged = merged[merged['filename'].isin(unique_filename_values)]

    output_dir =  os.path.dirname(all_data)
    os.makedirs(output_dir, exist_ok=True)

    merged.index.to_series().to_csv(os.path.join(output_dir, f"train_{filenames}.csv"), index=False, header=False)


def split_parquet(file, test_tasks=5, output_dir=""):
    df = pd.read_parquet(file)

    # Get unique tasks
    unique_tasks = df["task_id"].unique()

    # Ensure we have enough tasks
    total_tasks = len(unique_tasks)
    train_tasks = total_tasks - test_tasks

    if train_tasks <= 0:
        raise ValueError("Not enough unique tasks to perform the split.")

    # Split tasks
    tasks_train = unique_tasks[:train_tasks]
    tasks_test = unique_tasks[train_tasks:]

    # Create DataFrames
    df_train = df[df["task_id"].isin(tasks_train)]
    df_test = df[df["task_id"].isin(tasks_test)]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to Parquet files
    train_path = os.path.join(output_dir, "train_tasks.parquet")
    test_path = os.path.join(output_dir, "test_tasks.parquet")

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print(f"Files saved in {output_dir}:")
    print(f"- train_tasks.parquet ({len(df_train)} rows)")
    print(f"- test_tasks.parquet ({len(df_test)} rows)")

def get_loss_predictions_file(file):
    df = pd.read_csv(file, index_col=0)
    mae = median_absolute_error(df['label'], df['predictions'])
    return mae

def get_loss_predictions(dir):
    losses ={}
    for file in os.listdir(dir):
        filepath = os.path.join(dir, file)  # Full file path
        losses[file] = get_loss_predictions_file(filepath)
    return losses


def cross_validate(all_data, n_splits=5, output_dir="", random_state=42):
    """
    Perform group-based k-fold cross-validation and save fold indices.

    Parameters:
        all_data (str): Path to the CSV file containing the data
        n_splits (int): Number of folds for cross-validation
        output_dir (str): Directory to save the split indices
        random_state (int): Random seed for reproducibility
    """
    # Make sure output_dir ends with a slash if it's not empty
    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'

    # Read data
    df = pd.read_csv(all_data, index_col=0)
    df.index.to_series().to_csv(output_dir + "encoding_indices.csv", index=False, header=False)

    # Initialize the GroupKFold cross-validator
    group_kfold = GroupKFold(n_splits=n_splits)

    # Generate and save the fold indices
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(
            X=df.drop(columns="label"),
            y=df["label"],
            groups=df["filename"]
    )):
        # Save train and validation indices for this fold
        pd.DataFrame(train_idx, columns=['Index']).to_csv(
            output_dir + f"train_fold_{fold}.csv", index=False, header=False)
        pd.DataFrame(val_idx, columns=['Index']).to_csv(
            output_dir + f"val_fold_{fold}.csv", index=False, header=False)

    # Return information about the cross-validation setup
    return {
        'n_splits': n_splits,
        'total_samples': len(df),
        'samples_per_fold': len(df) // n_splits
    }
#cross_validate("../data/2.5M/all_data.csv",5,"../data/2.5M/cross_val")