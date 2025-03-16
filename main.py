from src.utils_data import create_dataset
import pandas as pd
#df = pd.read_parquet("raw_data/70_task.parquet", engine="pyarrow")
# tasks = df['task_id'].unique()[:10]
# df = df[df['task_id'].isin(tasks)]
# df = df.head(1000)
# print(df['filename'].nunique())
# print(len(df))
# df.to_parquet('raw_data/1k_rows.parquet', engine='pyarrow')
create_dataset("raw_data/1k_rows.parquet", "data/1k_rows/")