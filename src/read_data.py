import pandas as pd


def apply_index_file(data_df, i_file):
    df_i = pd.read_csv(open(i_file, "r"), index_col=False, header=None)
    return data_df.iloc[df_i[0]]



def read_train_val_test_data(args,lookup_df):
    """
    Read train val and/or test data. Reads all data dataframes for which the arguments were given, otherwise None
    :param args:
    :return:
    """
    if args.use_1_data_file:
        all_data = pd.read_csv(args.data_file, index_col=0)
        original_index = all_data.index
        all_data = pd.merge(all_data, lookup_df, on='filename',
                               how='left')  # You can adjust 'how' to 'left', 'right', or 'outer' based on your need
        all_data.index = original_index
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
        df_train = filter_on_tasks(
            (
                pd.read_csv(args.train_file, index_col=0)
                if args.train_file is not None
                else None
            ),
            args.tasks,
        )
        df_val = filter_on_tasks(
            (
                pd.read_csv(args.val_file, index_col=0)
                if args.val_file is not None
                else None
            ),
            args.tasks,
        )
        df_test = filter_on_tasks(
            (
                pd.read_csv(args.val_file, index_col=0)
                if args.test_file is not None
                else None
            ),
            args.tasks,
        )

    return df_train, df_val, df_test
