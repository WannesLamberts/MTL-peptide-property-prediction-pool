from src.utils_data import create_dataset,split_data
import pandas as pd
#df = pd.read_parquet("raw_data/70_task.parquet", engine="pyarrow")
# tasks = df['task_id'].unique()[:10]
# df = df[df['task_id'].isin(tasks)]
# df = df.head(1000)
# print(df['filename'].nunique())
# print(len(df))
# df.to_parquet('raw_data/1k_rows.parquet', engine='pyarrow')
#create_dataset("raw_data/70_task.parquet", "data/5_tasks/",20)
#create_dataset("raw_data/80_tasks.parquet", "data/10_filenames/",10,split_mode="filename")
#create_dataset2("data/80_tasks/80_tasks.parquet", "data/80_tasks/")

#create_dataset("raw_data/80_tasks.parquet", "data/200_filenames/all_data.csv",filter_filename=200)
#split_data("data/200_filenames/all_data.csv","data/200_filenames/filename=10,amount=0,split=filename/",filter_filename=10,split_mode="filename")
split_data("data/200_filenames/all_data.csv",split_mode="filename",filter_filename=200)

# df = pd.read_parquet("raw_data/80_tasks.parquet", engine="pyarrow")
# print(df[(df["sequence"]=="SASSSAAGSPGGLTSLQQQK") & (df["task_id"]=="04fe276206f54b3c8f81798ea92aa6e3")]["RT"])
