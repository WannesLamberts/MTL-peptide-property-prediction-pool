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
    x=data_series.values,
    y=data_series.index,
    alpha=0.8,
    color="#2ca02c"
    )
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
    colors = sns.color_palette("colorblind", len(data_series))
    ax = sns.barplot(
    x=data_series.index,
    y=data_series.values,
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

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()



def vertical_boxplot(data_series, ylabel, title, figsize=(6, 8)):

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

    plt.tight_layout()

    # Display the plot
    plt.show()


def horizontal_boxplot(data_series, xlabel, title, figsize=(8, 6),show_fliers=False):

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

    summary = pd.Series([s for s in data_series], index=labels)

    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





def plot_wasserstein_comparison(wasserstein_dict1, wasserstein_dict2,
                                label1="Dataset 1", label2="Dataset 2",
                                xlim=(0, 20), figsize=(10, 6)):

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
    sorted_data = sorted(zip(series.index, series.values), key=lambda x: x[1], reverse=False)
    categories_sorted, values_sorted = zip(*sorted_data)

    # Create figure with optimal dimensions
    fig, ax = plt.subplots(figsize=(10, 14))

    # Create horizontal bar plot
    bars = ax.barh(categories_sorted, values_sorted,
                   color='#95C5F9', alpha=0.8, edgecolor='white', linewidth=0.5)

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

    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust y-axis labels
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=10)

    # Set margins and layout
    ax.margins(y=0.01)  # Reduce vertical margins
    plt.tight_layout()



def horizontal_boxplot(data_series, xlabel, figsize=(8, 6), show_fliers=False):


    # Calculate statistics
    q1 = data_series.quantile(0.25)
    q2 = data_series.quantile(0.50)  # median
    q3 = data_series.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    # Identify outliers
    outliers = data_series[(data_series < lower_fence) | (data_series > upper_fence)]
    n_outliers = len(outliers)

    # Calculate whisker positions (actual min/max within fences)
    lower_whisker = data_series[data_series >= lower_fence].min()
    upper_whisker = data_series[data_series <= upper_fence].max()

    # Create figure
    plt.figure(figsize=figsize)

    # Create horizontal boxplot
    sns.boxplot(
        x=data_series.values,
        color="#95C5F9",
        linewidth=1.5,
        showfliers=show_fliers
    )

    # Customize labels
    plt.ylabel(xlabel, fontsize=12, labelpad=10)

    # Add grid
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Create legend text with statistics
    legend_text = [
        f"Q1 (25%): {q1:.2f}",
        f"Q2 (50%, Median): {q2:.2f}",
        f"Q3 (75%): {q3:.2f}",
        f"IQR: {iqr:.2f}",
        f"Lower Whisker: {lower_whisker:.2f}",
        f"Upper Whisker: {upper_whisker:.2f}",
        f"Outliers: {n_outliers}"
    ]

    # Add outlier range if outliers exist
    if n_outliers > 0:
        legend_text.append(f"Outlier Range: [{outliers.min():.2f}, {outliers.max():.2f}]")

    # Add legend
    plt.text(0.02, 0.98, '\n'.join(legend_text),
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
                                     title='',
                                     xlabel='Grouping Method', ylabel='MAD',
                                     bw=0.2, percentile_cutoff=0.95,lower=-0.1,figsize=(10, 6)):
    df_list = []
    for data, label in zip(data_series, labels):
        df_list.append(pd.DataFrame({
            'MAD': data.values if hasattr(data, 'values') else data,
            'Group': label
        }))
    df_violin = pd.concat(df_list, ignore_index=True)

    group_max_percentiles = []
    for group in df_violin['Group'].unique():
        group_data = df_violin[df_violin['Group'] == group]['MAD']
        group_95th = group_data.quantile(percentile_cutoff)
        group_max_percentiles.append(group_95th)

    cutoff = max(group_max_percentiles)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create vertical violin plot with a clean, light color
    sns.violinplot(x='Group', y='MAD', data=df_violin, cut=0, ax=ax,
                   inner='box', linewidth=1, bw=bw, color='#95C5F9')  # Light sky blue

    # Set y-axis limits (for MAD values)
    ax.set_ylim(lower, cutoff)

    # Annotate the cutoff line with a softer red
    ax.axhline(y=cutoff, color='#FF6B6B', linestyle='--', alpha=0.8, linewidth=1.2)

    # Calculate medians and quantile ranges for x-axis labels
    unique_groups = df_violin['Group'].unique()
    x_labels_with_medians = []
    for group in unique_groups:
        group_data = df_violin[df_violin['Group'] == group]['MAD']
        group_median = group_data.median()
        x_labels_with_medians.append(f"{group}\n(median: {group_median:.3f})")

    # Set custom x-axis labels with medians
    ax.set_xticklabels(x_labels_with_medians)

    # Set plot titles and labels
    ax.set_title(title, fontsize=14, color='#2C3E50')
    ax.set_xlabel(xlabel, fontsize=12, color='#34495E')
    ax.set_ylabel(ylabel, fontsize=12, color='#34495E')

    # Clean up the plot appearance
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFBFC')

    plt.tight_layout()
    plt.show()

def plot_scatter(data,y_label='value'):
    # Calculate quartiles and whiskers
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Calculate whisker positions (1.5 * IQR rule)
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create x-coordinates for scatterplot (you can modify this based on your needs)
    x_coords = np.arange(len(data))

    # Identify outliers and normal points
    below_lower_whisker = data < lower_whisker
    above_upper_whisker = data > upper_whisker
    normal_points = ~(below_lower_whisker | above_upper_whisker)

    if np.any(below_lower_whisker):
        ax.scatter(x_coords[below_lower_whisker], data[below_lower_whisker],
                   c='#95C5F9', s=50, label=f'Below lower whisker (<{lower_whisker:.2f}) n={len(below_lower_whisker)}',
                   marker='o', edgecolors='darkblue', linewidth=1)

    # Plot outliers above upper whisker in red
    if np.any(above_upper_whisker):
        ax.scatter(x_coords[above_upper_whisker], data[above_upper_whisker],
                   c='#95C5F9', s=50, label=f'Above upper whisker (>{upper_whisker:.2f}) n={len(above_upper_whisker)}',
                   marker='o', edgecolors='darkblue', linewidth=1)

    # Customize the plot
    ax.set_xlabel('Index')
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()

##########

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def combined_plot(data, xlabel="Count", figsize=(15, 10), range=(0, 40)):
    # Calculate statistics
    q1 = data.quantile(0.25)
    q2 = data.quantile(0.50)  # median
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # Identify outliers
    outliers = data[(data < lower_fence) | (data > upper_fence)]
    n_outliers = len(outliers)

    # Calculate whisker positions
    lower_whisker = data[data >= lower_fence].min()
    upper_whisker = data[data <= upper_fence].max()

    fig = plt.figure(figsize=figsize)

    ax1 = plt.subplot(2, 2, 1)  # Top left
    ax2 = plt.subplot(2, 2, 2)  # Top right
    ax3 = plt.subplot(2, 1, 2)  # Bottom row, spanning both columns

    # 1. Histogram
    ax1.hist(data.values, range=range, color="#95C5F9", edgecolor='black')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    # 2. Vertical Boxplot
    sns.boxplot(y=data.values, color="#95C5F9", linewidth=1.5, showfliers=False, ax=ax2)
    ax2.set_ylabel(xlabel)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add statistics text to boxplot
    legend_text = [
        f"Q1: {q1:.2f}",
        f"Median: {q2:.2f}",
        f"Q3: {q3:.2f}",
        f"IQR: {iqr:.2f}",
        f"Outliers: {n_outliers}"
    ]

    ax2.text(0.02, 0.98, '\n'.join(legend_text),
             transform=ax2.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    x_coords = np.arange(len(data))

    # Identify outlier positions
    below_lower_whisker = data < lower_fence
    above_upper_whisker = data > upper_fence

    # Count outliers
    below_count = np.sum(below_lower_whisker)
    above_count = np.sum(above_upper_whisker)

    # Plot outliers
    if np.any(below_lower_whisker):
        ax3.scatter(x_coords[below_lower_whisker], data[below_lower_whisker],
                    c='red', s=50, label=f'Below lower fence (<{lower_whisker})(n={below_count})',
                    marker='o', edgecolors='darkred', linewidth=1)

    if np.any(above_upper_whisker):
        ax3.scatter(x_coords[above_upper_whisker], data[above_upper_whisker],
                    c='#95C5F9', s=50, label=f'Above upper fence(>{upper_whisker}) (n={above_count})',
                    marker='o', edgecolors='darkblue', linewidth=1)
    ax3.set_xlabel('Index')
    ax3.set_ylabel(xlabel)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def combined_plot_no_box(data, xlabel="Count", figsize=(15, 6), range=(0, 40)):


    # Calculate statistics
    q1 = data.quantile(0.25)
    q2 = data.quantile(0.50)  # median
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # Identify outliers
    outliers = data[(data < lower_fence) | (data > upper_fence)]
    n_outliers = len(outliers)

    # Calculate whisker positions
    lower_whisker = data[data >= lower_fence].min()
    upper_whisker = data[data <= upper_fence].max()

    # Create figure with subplots in a 1x2 grid (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Histogram (Barplot)
    ax1.hist(data.values, range=range, color="#95C5F9", edgecolor='black')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Frequency")
    ax1.set_title("Histogram")
    ax1.grid(True, alpha=0.3)

    # Add statistics text to histogram
    legend_text = [
        f"Q1: {q1:.2f}",
        f"Median: {q2:.2f}",
        f"Q3: {q3:.2f}",
        f"IQR: {iqr:.2f}",
        f"Outliers: {n_outliers}"
    ]

    ax1.text(0.02, 0.98, '\n'.join(legend_text),
             transform=ax1.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Scatter plot for outliers
    x_coords = np.arange(len(data))

    # Identify outlier positions
    below_lower_whisker = data < lower_fence
    above_upper_whisker = data > upper_fence

    # Count outliers
    below_count = np.sum(below_lower_whisker)
    above_count = np.sum(above_upper_whisker)

    # Plot all data points first (optional - for context)
    ax2.scatter(x_coords, data, c='lightgray', s=20, alpha=0.5, label='Normal values')

    # Plot outliers on top
    if np.any(below_lower_whisker):
        ax2.scatter(x_coords[below_lower_whisker], data[below_lower_whisker],
                    c='red', s=50, label=f'Below lower fence (<{lower_whisker:.2f}) (n={below_count})',
                    marker='o', edgecolors='darkred', linewidth=1)

    if np.any(above_upper_whisker):
        ax2.scatter(x_coords[above_upper_whisker], data[above_upper_whisker],
                    c='#95C5F9', s=50, label=f'Above upper fence (>{upper_whisker:.2f}) (n={above_count})',
                    marker='o', edgecolors='darkblue', linewidth=1)

    ax2.set_xlabel('Index')
    ax2.set_ylabel(xlabel)
    ax2.set_title("Outlier Analysis")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def simple_violin_plot(data_series, labels, title="Violin Plot", width=0.8,
                       cutoff_percentile_upper=None, cutoff_percentile_lower=None,
                       zero_line=False, colors=['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon',
                                                'lightsteelblue', 'lightpink', 'lightgray', 'lightcyan']):
    all_data = np.concatenate(data_series)

    if cutoff_percentile_upper is None and cutoff_percentile_lower is None:
        y_min = np.min(all_data)
        y_max = np.max(all_data)
        zoom_applied = False
    else:
        if cutoff_percentile_lower is not None:
            y_min = np.percentile(all_data, cutoff_percentile_lower)
            y_max = np.percentile(all_data, cutoff_percentile_upper) if cutoff_percentile_upper is not None else np.max(
                all_data)
        else:
            y_min = np.min(all_data)
            y_max = np.percentile(all_data, cutoff_percentile_upper) if cutoff_percentile_upper is not None else np.max(
                all_data)
        zoom_applied = True

    data_range = y_max - y_min
    buffer = data_range * 0.05
    y_min = y_min - buffer
    y_max = y_max + buffer

    plt.figure(figsize=(6, 6))

    all_positive = all(np.min(series) >= 0 for series in data_series)
    cut_param = 0 if all_positive else 2

    ax = sns.violinplot(data=data_series, width=width, cut=cut_param,
                        palette=colors[:len(data_series)])

    enhanced_labels = []
    for i, (label, original_data) in enumerate(zip(labels, data_series)):
        median_val = np.median(original_data)
        enhanced_labels.append(f"{label}\n(med: {median_val:.2f})")

    ax.set_xticklabels(enhanced_labels)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind the violins

    if zero_line:
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()