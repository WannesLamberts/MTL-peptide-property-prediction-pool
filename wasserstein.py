

import pandas as pd
from scipy.stats import wasserstein_distance
import itertools

from joblib import Parallel, delayed
import itertools
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from src.plot import plot_wasserstein_comparison


def compute_wasserstein_for_sequence(sequence, group, min_files, col='filename'):
    distributions = group.groupby(col)['iRT'].apply(list)
    distributions_dict = {k: v for k, v in distributions.to_dict().items() if len(v) >= min_files}
    filenames = list(distributions_dict.keys())

    if len(filenames) < 2:  # Need at least 2 files to compute distance
        return sequence, None

    distances = [
        wasserstein_distance(distributions_dict[file1], distributions_dict[file2])
        for file1, file2 in itertools.combinations(filenames, 2)
    ]

    if distances:
        return sequence, sum(distances) / len(distances)
    return sequence, None


def compute_mean_wasserstein_distance(df, min_files, col='filename', n_jobs=19):
    grouped = df.groupby('sequence')
    groups = list(grouped)  # Convert to list for tqdm compatibility

    # Parallel execution with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_wasserstein_for_sequence)(sequence, group, min_files, col)
        for sequence, group in tqdm(groups, desc="Processing sequences")
    )

    # Filter out None results and create dictionary
    mean_distances = {seq: dist for seq, dist in results if dist is not None}

    return mean_distances
if __name__ == "__main__":
    DATASET_2 = "raw_data/proteome.parquet"
    df2 = pd.read_parquet(DATASET_2)
    mean_wasserstein2 = compute_mean_wasserstein_distance(df2, 1, 'pool')
    plot_wasserstein_comparison(
        mean_wasserstein2,
        mean_wasserstein2,
        label1="MassiveKB",
        label2="Proteome"
    )


