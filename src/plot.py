import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_bar_horizontal(data_series,title="Sequences by dataset for split",
                        xlabel="Count of Sequences", ylabel="Dataset",
                        figsize=(10, 6),sort=True):
    if sort:
        data_series = data_series.sort_values(ascending=False)

    plt.figure(figsize=figsize)
    ax = sns.barplot(
    x=data_series.values,  # Values of the Series (counts)
    y=data_series.index,   # Index of the Series (datasets)
    alpha=0.8,
    color="#2ca02c"
    )
        # Add value labels to the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)

    # Customize the plot
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    # Remove spines
    sns.despine()

    # Add grid lines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_bar_vertical(data_series,title,xlabel,ylabel,figsize=(10, 6),sort=True):
    if sort:
        data_series = data_series.sort_values(ascending=False)
    plt.figure(figsize=figsize)
    colors = sns.color_palette("colorblind", len(data_series))  # You can use other palettes too
    ax = sns.barplot(
    x=data_series.index,  # Values of the Series (counts)
    y=data_series.values,   # Index of the Series (datasets)
    palette=colors,
    alpha=0.8,
    )
        # Add value labels to the bars
    for container in ax.containers:
        ax.bar_label(container, padding=3)

    # Customize the plot
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    # Remove spines
    sns.despine()

    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

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