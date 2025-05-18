import pandas as pd


def apply_index_file(data_df, i_file):
    df_i = pd.read_csv(open(i_file, "r"), index_col=False, header=None)
    return data_df.iloc[df_i[0]]

def read_train_val_test(args,lookup_df):
    all_data = pd.read_parquet(args.data_file)
    train_indices = pd.read_csv(args.train_i) if args.train_i is not None else None
    df_train = apply_index_file(all_data, args.train_i)
    df_train = pd.merge(df_train, lookup_df, on='filename',
                        how='left')  # You can adjust 'how' to 'left', 'right', or 'outer' based on your need
    df_val = pd.read_parquet(args.val_file) if args.val_file is not None else None
    df_val = pd.merge(df_val, lookup_df, on='filename',
                      how='left')  # You can adjust 'how' to 'left', 'right', or 'outer' based on your need
    if args.test_file is not None:
        df_test = pd.read_parquet(args.test_file)
        df_test = pd.merge(df_test, lookup_df, on='filename', how='left')
    else:
        df_test = None

    return df_train, df_val, df_test

def read_train_val_test_data(args,lookup_df=None):
    """
    Read train val and/or test data. Reads all data dataframes for which the arguments were given, otherwise None
    :param args:
    :return:
    """
    if args.use_1_data_file:
        all_data = pd.read_parquet(args.data_file)
        if args.type=="pool":
            all_data = pd.merge(all_data, lookup_df, on='filename',
                                   how='left')  # You can adjust 'how' to 'left', 'right', or 'outer' based on your need
        df_train = (
            apply_index_file(all_data, args.train_i)
            if args.train_i is not None
            else None
        )
        df_val = (apply_index_file(all_data, args.val_i)
            if args.val_i is not None
            else None
        )
        df_test = (apply_index_file(all_data, args.test_i)
            if args.test_i is not None
            else None
        )

    else:
        df_train = pd.read_parquet(args.train_file) if args.train_file is not None else None
        df_train = pd.merge(df_train, lookup_df, on='filename',
                               how='left')  # You can adjust 'how' to 'left', 'right', or 'outer' based on your need
        df_val = pd.read_parquet(args.val_file) if args.val_file is not None else None
        df_val = pd.merge(df_val, lookup_df, on='filename',
                               how='left')  # You can adjust 'how' to 'left', 'right', or 'outer' based on your need
        if args.test_file is not None:
            df_test = pd.read_parquet(args.test_file)
            df_test = pd.merge(df_test, lookup_df, on='filename', how='left')
        else:
            df_test = None

    return df_train, df_val, df_test
