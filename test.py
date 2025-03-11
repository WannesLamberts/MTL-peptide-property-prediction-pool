import pandas as pd
from sklearn.model_selection import train_test_split


def create_model_dataset(file,output):
    df = pd.read_csv(file, sep='\t', index_col=False)
    # Select relevant columns
    df = df[['sequence', 'iRT','filename']]

    # Rename columns correctly
    df.columns = ['modified_sequence', 'label','filename']

    # Add missing columns
    df['task'] = 'iRT'
    df['Charge'] = ''  # Empty values
    df['DCCS_sequence'] = ''  # Empty values

    df.to_csv(output, index=True)


def split_data(file, train_ratio, val_ratio, test_ratio,output_dir):
    # Load the dataset
    df = pd.read_csv(file, sep='\t')

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
filename = "og_data/test_data_10_filenames.tsv"
create_model_dataset(filename,"data/sample_10_filenames/all_data.csv")

split_data('data/sample_10_filenames/all_data.csv',0.8,0.1,0.1,'data/sample_10_filenames/')













# df = pd.read_csv("og_data/test_data_calibrated_merged.tsv", delimiter="\t")
#
# # Select 10 random filenames from the 'filename' column
# selected_filenames = df['filename'].sample(n=10, random_state=42).tolist()
#
# # Create a new DataFrame with only rows where 'filename' is in the selected filenames
# filtered_df = df[df['filename'].isin(selected_filenames)]
#
# # Display the filtered DataFrame
# print(filtered_df)
# filtered_df.to_csv("og_data/test_data_10_filenames.tsv", sep="\t", index=False)
