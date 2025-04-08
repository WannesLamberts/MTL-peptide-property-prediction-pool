import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def barplot(data_series, xlabel,ylabel, title, figsize=(12, 6)):
    """
    Generate a barplot for a given pandas Series.

    Parameters:
    - data_series: pandas Series with index (e.g., dataset names) and values (e.g., counts)
    - xlabel: str, label for x-axis
    - title: str, title of the plot
    - figsize: tuple, figure size (width, height)
    """
    # Sort the dataset
    sorted_data = data_series.sort_values(ascending=False)

    # Create figure
    plt.figure(figsize=figsize)

    # Create barplot
    bars = sns.barplot(
        x=sorted_data.values,
        y=sorted_data.index,
        hue=sorted_data.index,
        palette="viridis",
        edgecolor="black",
        linewidth=0.5,
        legend=False
    )

    # Add value labels to the right of bars
    for i, v in enumerate(sorted_data.values):
        plt.text(v, i, f' {v:,}',
                 va='center',
                 ha='left',
                 fontsize=10)

    # Customize labels and title
    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=15)

    # Add simple grid
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Tight layout
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_sequences_split(df1, df2, df3, title="Sequences by dataset for split",
                         xlabel="Count of Sequences", ylabel="Dataset",
                         figsize=(10, 6), sort_by_count=True, use_unique=True):
    """
    Create a horizontal bar plot showing the count of sequences
    (unique or total) for each dataset across three dataframes.

    Parameters:
    ----------
    df1, df2, df3 : pandas DataFrames
        The three dataframes to compare
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size (width, height)
    sort_by_count : bool
        If True, sorts datasets by total count across all sources
    use_unique : bool
        If True, counts unique sequences; otherwise, counts total occurrences
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Label each dataframe with its source
    df1 = df1.copy()
    df2 = df2.copy()
    df3 = df3.copy()

    df1['source'] = 'train'
    df2['source'] = 'validation'
    df3['source'] = 'test'

    # Combine the dataframes
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)

    # Choose aggregation function
    agg_func = 'nunique' if use_unique else 'count'

    # Count sequences for each dataset in each source
    sequence_counts = combined_df.groupby(['source', 'dataset'])['sequence'].agg(agg_func).reset_index(name='count')

    # Sort the data if requested
    if sort_by_count:
        # Calculate total counts for each dataset across all sources
        total_counts = sequence_counts.groupby('dataset')['count'].sum().reset_index()
        total_counts = total_counts.sort_values('count', ascending=False)

        # Create a categorical type with the ordered datasets
        ordered_datasets = pd.CategoricalDtype(categories=total_counts['dataset'].tolist(), ordered=True)

        # Convert dataset column to the ordered categorical type
        sequence_counts['dataset'] = sequence_counts['dataset'].astype(ordered_datasets)

    # Create the horizontal bar plot
    plt.figure(figsize=figsize)

    # Use a colorblind-friendly palette
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']

    ax = sns.barplot(
        x='count',
        y='dataset',
        hue='source',
        data=sequence_counts,
        palette=colors,
        alpha=0.8
    )

    # Add value labels to the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)

    # Customize the plot
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Move legend to a better position
    plt.legend(title="split", bbox_to_anchor=(1, 1), loc='upper left')

    # Remove spines
    sns.despine()

    # Add grid lines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return plt


def vertical_boxplot(data_series, ylabel, title, figsize=(6, 8)):
    """
    Generate a vertical boxplot for a given pandas Series.

    Parameters:
    - data_series: pandas Series with values (e.g., counts)
    - ylabel: str, label for y-axis
    - title: str, title of the plot
    - figsize: tuple, figure size (width, height)
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Create vertical boxplot
    sns.boxplot(
        y=data_series.values,
        color="lightblue",
        linewidth=1.5,
        fliersize=5
    )

    # Customize labels and title
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=15)

    # Add grid
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Tight layout
    plt.tight_layout()

    # Display the plot
    plt.show()