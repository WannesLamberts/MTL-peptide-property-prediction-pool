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


def horizontal_boxplot(data_series, xlabel, title, figsize=(8, 6)):
    """
    Generate a horizontal boxplot for a given pandas Series.

    Parameters:
    - data_series: pandas Series with values (e.g., counts)
    - xlabel: str, label for x-axis
    - title: str, title of the plot
    - figsize: tuple, figure size (width, height)
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Create horizontal boxplot
    sns.boxplot(
        x=data_series.values,  # Changed from y to x
        color="lightblue",
        linewidth=1.5,
        fliersize=5
    )

    # Customize labels and title
    plt.xlabel(xlabel, fontsize=12, labelpad=10)  # Changed ylabel to xlabel
    plt.title(title, fontsize=14, pad=15)

    # Add grid
    plt.grid(axis="x", linestyle="--", alpha=0.7)  # Changed grid axis to x

    # Tight layout
    plt.tight_layout()

    # Display the plot
    plt.show()


def create_MAD_comparison_boxplot(data_series, labels, title="Comparison Boxplot",
                              xlabel="Grouping Method", ylabel="Value",
                              show_fliers=False):
    df_list = []
    for i, (data, label) in enumerate(zip(data_series, labels)):
        df_list.append(pd.DataFrame({
            ylabel: data.values if hasattr(data, 'values') else data,
            'Group': label
        }))

    df_boxplot = pd.concat(df_list, ignore_index=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='Group', y=ylabel, data=df_boxplot, showfliers=show_fliers, ax=ax)

    # Set plot labels and title
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Optional: Create a summary Series similar to the original code
    summary = pd.Series([s for s in data_series], index=labels)

    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_MAD_comparison_violinplot(data_series, labels,
                                     title='Comparison of MAD values across different groupings',
                                     xlabel='Grouping Method', ylabel='MAD',
                                     bw=0.2, percentile_cutoff=0.95):
    # Build the combined DataFrame
    df_list = []
    for data, label in zip(data_series, labels):
        df_list.append(pd.DataFrame({
            ylabel: data.values if hasattr(data, 'values') else data,
            'Group': label
        }))

    df_violin = pd.concat(df_list, ignore_index=True)

    # Calculate cutoff (e.g., 95th percentile)
    cutoff = df_violin[ylabel].quantile(percentile_cutoff)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create violin plot
    sns.violinplot(x='Group', y=ylabel, data=df_violin, cut=0, ax=ax,
                   inner='box', linewidth=1, bw=bw)

    # Set y-axis limits
    ax.set_ylim(-1, cutoff)

    # Annotate the cutoff line
    ax.axhline(y=cutoff, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.text(0.02, cutoff * 0.95, f"Cutoff at {cutoff:.2f} ({int(percentile_cutoff * 100)}th percentile)",
            ha='left', va='top', transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

    # Annotate max values above cutoff
    unique_groups = df_violin['Group'].unique()
    for i, group in enumerate(unique_groups):
        group_max = df_violin[df_violin['Group'] == group][ylabel].max()
        if group_max > cutoff:
            ax.annotate(f"Max: {group_max:.2f}",
                        xy=(i, cutoff), xytext=(i, cutoff * 0.85),
                        ha='center', va='top',
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    # Set plot titles and labels
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()
