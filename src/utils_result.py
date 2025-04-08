import os
import glob
import pandas as pd
from sklearn.metrics import median_absolute_error


def get_predictions_from_logs(log_dir, config_x=None):
    predictions = []

    # Iterate over directories in log_dir
    for sub_dir in os.listdir(log_dir):
        # If config_x is specified, filter directories that start with CONFIG=X
        if config_x is None or sub_dir.startswith(f"CONFIG={config_x}"):
            version_path = os.path.join(log_dir, sub_dir, "version_0", "predictions")

            if os.path.isdir(version_path):  # Ensure it's a valid directory
                # Find all CSV files matching the naming pattern
                csv_files = glob.glob(os.path.join(version_path, "predict_loss=*.csv"))
                predictions.extend(csv_files)

    return predictions


def get_loss_predictions_file(file):
    df = pd.read_csv(file, index_col=0)
    mae = median_absolute_error(df['label'], df['predictions'])
    return mae

def get_loss_predictions(files):
    losses ={}
    for file in files:
        losses[file] = get_loss_predictions_file(file)
    return losses
