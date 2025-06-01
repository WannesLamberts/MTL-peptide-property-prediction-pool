from ctypes import pydll


from src.utils_data import *
import pandas as pd
stri="aabbccdee"
prev=""
c=1
for x in stri:
   if x!=prev and c==0:
       print("non repeating char is "+prev)
       break
   elif x!=prev and c==1:
       c=0
       prev=x
   else:
       c=1
       prev=x





import pandas as pd
#
# df_1 = pd.read_parquet("raw_data/80_tasks.parquet")
# df_2 = pd.read_parquet("raw_data/76_tasks.parquet")
# # Concatenate the DataFrames
# df_combined = pd.concat([df_1, df_2], ignore_index=True)
#
# # Write to a new Parquet file
# df_combined.to_parquet("raw_data/combined_tasks.parquet", index=False)

# df2 = pd.read_parquet("data/low_variety/train_low_variety.parquet")

# p1 = df1["modified_sequence"].unique()
# p2 = df2["modified_sequence"].unique()
# # Convert to sets
# set1 = set(p1)
# set2 = set(p2)

# Find overlap (intersection)
# overlap = set1 & set2  # or set1.intersection(set2)
#
# # See overlap
# print(overlap)
# Convert to DataFrame
# df = pd.read_parquet("data/parquet/train_low_variety.parquet")
# # Select relevant columns
# df = df[['sequence', 'iRT', 'filename']]
#
# # Rename columns correctly
# df.columns = ['modified_sequence', 'label', 'filename']

# Add missing columns
# df['task'] = 'iRT'
# df_train_practice = df.iloc[:1000]
# df_val_practice = df.iloc[1001:1100]
# df_test_practice = df.iloc[1101:1200]

# Save as Parquet
# df_train_practice.to_parquet("data/parquet/practice_train.parquet", index=False)
# df_val_practice.to_parquet("data/parquet/practice_val.parquet", index=False)
# df_test_practice.to_parquet("data/parquet/practice_test.parquet", index=False)

#split_parquet("raw_data/80_tasks.parquet",output_dir="raw_data")
# df = pd.read_parquet("raw_data/test_tasks.parquet",engine="pyarrow")
# g = df.groupby(['task_id','dataset']).size()
# print(g)
# print(df["dataset"].unique())
# df2 = pd.read_parquet("raw_data/80_tasks.parquet",engine="pyarrow")
# print(df2["dataset"].unique())
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



