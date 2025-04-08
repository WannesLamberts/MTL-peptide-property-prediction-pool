from ctypes import pydll


from src.utils_data import *
import pandas as pd


#split_parquet("raw_data/80_tasks.parquet",output_dir="raw_data")
df = pd.read_parquet("raw_data/test_tasks.parquet",engine="pyarrow")
g = df.groupby(['task_id','dataset']).size()
print(g)
print(df["dataset"].unique())
df2 = pd.read_parquet("raw_data/80_tasks.parquet",engine="pyarrow")
print(df2["dataset"].unique())
# create_dataset("raw_data/train_tasks.parquet", "data/trial/all_data.csv",10)
# split_data("data/trial/all_data.csv",output_dir="data/trial/")
# select_train_data("data/200_filenames/all_data.csv","data/200_filenames/train.csv",200)


# create_dataset("raw_data/train_tasks.parquet", "data/2.5M/all_data.csv",amount=0.09)
# split_data("data/2.5M/all_data.csv",output_dir="data/2.5M/")
#select_train_data("data/2.5M/all_data.csv","data/2.5M/train.csv",1151)
# test = pd.read_csv("data/2.5M/all_data.csv",index_col=64)
# group = test.groupby('filename')
# print(len(group))

# test = pd.read_csv("data/2.5M/all_data.csv",index_col=0)
# group = test.groupby('filename')["modified_sequence"].count()
# print(group)
# df = pd.read_parquet("raw_data/80_tasks.parquet",engine="pyarrow")
# print(len(df))
# create_dataset("raw_data/train_tasks.parquet", "data/2.5M/all_data.csv",amount=2500000)

# df = pd.read_parquet("raw_data/test_tasks.parquet",engine="pyarrow")
# create_dataset("raw_data/test_tasks.parquet", "data/test_50/all_data.csv",amount=0.0701611)


# df.index.to_series().to_csv(os.path.join("data/test_50/", "test.csv"), index=False, header=False)
# df = pd.read_csv("data/2.5M/all_data.csv")
# uniq = df.groupby("filename")
# print(len(uniq))



