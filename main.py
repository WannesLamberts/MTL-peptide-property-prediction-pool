from src.utils_data import create_dataset
import pandas as pd
#df = pd.read_parquet("raw_data/70_task.parquet", engine="pyarrow")
# tasks = df['task_id'].unique()[:10]
# df = df[df['task_id'].isin(tasks)]
# df = df.head(1000)
# print(df['filename'].nunique())
# print(len(df))
# df.to_parquet('raw_data/1k_rows.parquet', engine='pyarrow')
# create_dataset("raw_data/70_task.parquet", "data/5_tasks/",5)
import torch

# Given corrected data
data = [(torch.tensor([[-0.0119], [-0.1]], dtype=torch.float16), [2, 4]),
        (torch.tensor([[-0.0658], [-3.0]], dtype=torch.float16), [3, 8])]

# Combine the tensors
combined_tensor = torch.cat([item[0] for item in data], dim=0)

# Combine the lists
combined_list = [item[1] for item in data]

# Flatten the list of lists
combined_list = [elem for sublist in combined_list for elem in sublist]
print(combined_list)
# Output
combined_tensor, combined_list