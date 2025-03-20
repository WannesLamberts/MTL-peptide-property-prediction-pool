import pandas as pd
from sklearn.model_selection import train_test_split
import os


def create_dataset_encoding(df):

    # Select relevant columns
    df = df[['sequence', 'iRT']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label']

    df['Charge'] = ''  # Empty values
    df['DCCS_sequence'] = ''  # Empty values
    return df

def create_dataset(file, output_dir, filter_filename=None,split_mode=None,amount=None):
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
    print(len(df))
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    if amount:
        df = df.head(amount)
    df.to_csv(output_dir + "all_data.csv", index=True)
    if split_mode == "regular":
        split_data_regular(df, 0.8, 0.1, 0.1, output_dir)
    elif split_mode =="filename":
        split_data_filename(df, 0.8, 0.1, 0.1, output_dir)


def split_data_filename(df, train_ratio, val_ratio, test_ratio,output_dir):
    unique_filenames = df['filename'].unique()
    print(len(unique_filenames))
    # Split into train and temp (val + test)
    train_filenames, temp_filenames = train_test_split(
        unique_filenames, test_size=(val_ratio + test_ratio), random_state=42)

    # Split temp into validation and test
    val_filenames, test_filenames = train_test_split(
        temp_filenames, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    # Get indices for each split
    train_indices = df[df['filename'].isin(train_filenames)].index.to_numpy()
    val_indices = df[df['filename'].isin(val_filenames)].index.to_numpy()
    test_indices = df[df['filename'].isin(test_filenames)].index.to_numpy()

    # Save indices to CSV files
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(train_indices).to_csv(os.path.join(output_dir, "train_0.csv"), index=False, header=False)
    pd.DataFrame(val_indices).to_csv(os.path.join(output_dir, "val_0.csv"), index=False, header=False)
    pd.DataFrame(test_indices).to_csv(os.path.join(output_dir, "test_0.csv"), index=False, header=False)

    return train_indices, val_indices, test_indices


def split_data_regular(df, train_ratio, val_ratio, test_ratio,output_dir):


    # Get indices before splitting
    indices = df.index.to_numpy()

    # Split into training and temp (val + test)
    train_indices, temp_indices = train_test_split(indices, test_size=(1 - train_ratio), random_state=42)

    # Calculate the proportion of validation and test relative to temp
    val_size = val_ratio / (val_ratio + test_ratio)  # Proportion for validation

    # Split temp into validation and test sets
    val_indices, test_indices = train_test_split(temp_indices, test_size=(1 - val_size), random_state=42)

    # Save indices to CSV files
    pd.DataFrame(train_indices, columns=['Index']).to_csv(output_dir+"train_0.csv", index=False, header=False)
    pd.DataFrame(val_indices, columns=['Index']).to_csv(output_dir+"val_0.csv", index=False, header=False)
    pd.DataFrame(test_indices, columns=['Index']).to_csv(output_dir+"test_0.csv", index=False, header=False)

    return train_indices, val_indices, test_indices