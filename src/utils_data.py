import pandas as pd
from sklearn.model_selection import train_test_split
import os


def create_dataset(file, output_dir, filter_filename=None):
    df = pd.read_parquet(file, engine="pyarrow")

    # Select relevant columns
    df = df[['sequence', 'iRT', 'filename']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label', 'filename']

    # Add missing columns
    df['task'] = 'iRT'

    # Filter to only keep the first 50 unique values of x if option is enabled
    if filter_filename:
        unique_filename_values = df['filename'].unique()[:filter_filename]
        df = df[df['filename'].isin(unique_filename_values)]

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    df.to_csv(output_dir + "all_data.csv", index=True)

    split_data(df, 0.8, 0.1, 0.1, output_dir)


def create_dataset_encoding(df):

    # Select relevant columns
    df = df[['sequence', 'iRT']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label']

    df['Charge'] = ''  # Empty values
    df['DCCS_sequence'] = ''  # Empty values
    return df

def split_data(df, train_ratio, val_ratio, test_ratio,output_dir):


    # Ensure the sum of ratios is 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

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


#create_dataset("../dataset.parquet", "data/par/")