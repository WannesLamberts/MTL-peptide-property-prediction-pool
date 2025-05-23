import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def horizontal_boxplot(data_series, xlabel, title, figsize=(8, 6),show_fliers=False):
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
        showfliers=show_fliers
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





def plot_wasserstein_comparison(wasserstein_dict1, wasserstein_dict2,
                                label1="Dataset 1", label2="Dataset 2",
                                xlim=(0, 20), figsize=(10, 6)):
    """
    Create KDE plot comparing two sets of Wasserstein distances.

    Args:
        wasserstein_dict1: First dictionary of Wasserstein distances
        wasserstein_dict2: Second dictionary of Wasserstein distances
        label1: Label for the first dataset
        label2: Label for the second dataset
        xlim: x-axis limits as tuple (min, max)
        figsize: Figure size as tuple (width, height)
    """
    # Extract values
    x_values1 = np.array(list(wasserstein_dict1.values()))
    x_values2 = np.array(list(wasserstein_dict2.values()))

    # Create KDE plot
    plt.figure(figsize=figsize)

    # Plot both distributions
    sns.kdeplot(x_values1, fill=True, alpha=0.5, color="skyblue", label=label1)
    sns.kdeplot(x_values2, fill=True, alpha=0.5, color="orange", label=label2)

    # Add vertical lines for medians
    mean1 = np.mean(x_values1)
    mean2 = np.mean(x_values2)
    median1 = np.median(x_values1)
    median2 = np.median(x_values2)

    plt.axvline(median1, color='blue', linestyle='--',
                label=f"{label1} Median: {median1:.2f}")
    plt.axvline(median2, color='red', linestyle='--',
                label=f"{label2} Median: {median2:.2f}")

    # Set plot limits and labels
    plt.xlim(xlim)
    plt.xlabel("Wasserstein Distance")
    plt.ylabel("Density")
    plt.title("Comparison of Wasserstein Distances")
    plt.legend()

    # Show statistics
    print(f"Statistics for {label1}:")
    print(f"  Mean: {mean1:.2f}")
    print(f"  Median: {median1:.2f}")
    print(f"  Min: {np.min(x_values1):.2f}")
    print(f"  Max: {np.max(x_values1):.2f}")
    print(f"  Count: {len(x_values1)}")

    print(f"\nStatistics for {label2}:")
    print(f"  Mean: {mean2:.2f}")
    print(f"  Median: {median2:.2f}")
    print(f"  Min: {np.min(x_values2):.2f}")
    print(f"  Max: {np.max(x_values2):.2f}")
    print(f"  Count: {len(x_values2)}")

    # Show the plot
    plt.tight_layout()
    plt.show()

    return plt


######

def plot_dataset_distributions(series,x_label,y_label):
    # Sort data by values for better visualization (optional)
    sorted_data = sorted(zip(series.index, series.values), key=lambda x: x[1], reverse=False)
    categories_sorted, values_sorted = zip(*sorted_data)

    # Create figure with optimal dimensions
    fig, ax = plt.subplots(figsize=(10, 14))

    # Create horizontal bar plot
    bars = ax.barh(categories_sorted, values_sorted,
                   color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.5)

    # Customize the plot
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values_sorted)):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2,
                f'{value}', va='center', ha='left', fontsize=8, color='black')

    # Styling improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Add subtle grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust y-axis labels
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=10)

    # Set margins and layout
    ax.margins(y=0.01)  # Reduce vertical margins
    plt.tight_layout()


def horizontal_boxplot(data_series, xlabel, figsize=(8, 6),show_fliers=False):
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
        showfliers=show_fliers
    )

    # Customize labels and title
    plt.xlabel(xlabel, fontsize=12, labelpad=10)  # Changed ylabel to xlabel

    # Add grid
    plt.grid(axis="x", linestyle="--", alpha=0.7)  # Changed grid axis to x

    # Tight layout
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_kde_grouped(df, threshold,sequence_col="modified_sequence",run_col="filename",label_name="label", legend=None):
    grouped = df.groupby(sequence_col)
    for sequence, seq_group in grouped:
        plt.figure(figsize=(8, 6))
        # Group again within each sequence by filename to get iRT values
        for filename, values in seq_group.groupby(run_col)[label_name]:
            values = list(values)  # Convert Series to list
            if len(values) >= threshold:
                label = f"{filename} (n={len(values)})"
                sns.kdeplot(values, label=label, fill=False)

        plt.xlabel("Retention Time")
        plt.ylabel("Density")
        if legend:
            plt.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"KDE Plot for Sequence: {sequence}")
        plt.show()

def create_MAD_comparison_violinplot(data_series, labels,
                                     title='Comparison of MAD values across different groupings',
                                     xlabel='MAD', ylabel='Grouping Method',
                                     bw=0.2, percentile_cutoff=0.95):
    # Build the combined DataFrame
    df_list = []
    for data, label in zip(data_series, labels):
        df_list.append(pd.DataFrame({
            'MAD': data.values if hasattr(data, 'values') else data,
            'Group': label
        }))

    df_violin = pd.concat(df_list, ignore_index=True)

    # Calculate cutoff (e.g., 95th percentile)
    cutoff = df_violin['MAD'].quantile(percentile_cutoff)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create horizontal violin plot with a vibrant color
    sns.violinplot(x='MAD', y='Group', data=df_violin, cut=0, ax=ax,
                   inner='box', linewidth=1, bw=bw, color='#4C78A8')  # Deep blue color

    # Set x-axis limits (for MAD values)
    ax.set_xlim(-0.1, cutoff)

    # Annotate the cutoff line (vertical now)
    ax.axvline(x=cutoff, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.text(cutoff * 0.95, 0.02, f"Cutoff at {cutoff:.2f} ({int(percentile_cutoff * 100)}th percentile)",
            ha='right', va='bottom', transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

    # Annotate max values beyond cutoff
    unique_groups = df_violin['Group'].unique()
    for i, group in enumerate(unique_groups):
        group_max = df_violin[df_violin['Group'] == group]['MAD'].max()
        if group_max > cutoff:
            ax.annotate(f"Max: {group_max:.2f}",
                        xy=(cutoff, i), xytext=(cutoff * 0.85, i),
                        ha='right', va='center',
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    # Add centered median annotations slightly below the line
    for i, group in enumerate(unique_groups):
        group_median = df_violin[df_violin['Group'] == group]['MAD'].median()
        ax.annotate(f"Median: {group_median:.2f}",
                    xy=(group_median, i), xytext=(group_median, i - 0.15),  # Shift down by 0.15
                    ha='center', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # Set plot titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()