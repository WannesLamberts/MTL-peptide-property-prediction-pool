import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
import os
from sklearn.metrics import median_absolute_error,median_absolute_error

def create_dataset(file, out_file):
    df = pd.read_parquet(file, engine="pyarrow")
    # Select relevant columns
    df = df[['sequence', 'iRT', 'filename']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label', 'filename']

    # Add missing columns
    df['task'] = 'iRT'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    df = df.reset_index(drop=True)
    df.to_parquet(out_file, index=True)


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
