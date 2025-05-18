import random
import re

import torch
from torch.utils.data import Dataset, default_collate

from src.util import end_padding


class MTLPepDataset(Dataset):
    def __init__(self, df, args):
        self.df = df.sample(frac=1)
        self.args = args
        self.pep_col = "modified_sequence"

        self._replace_mods()

    def _replace_mods(self):
        for mod, letter in self.args.vocab.modifications.items():
            if mod.startswith("_"):
                self.df[self.pep_col] = self.df[self.pep_col].str.replace(
                    "^" + re.escape(mod[1:]), letter, regex=True
                )
            else:
                self.df[self.pep_col] = self.df[self.pep_col].str.replace(
                    mod, letter
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        peptide = self.df[self.pep_col].iloc[item]
        ids = self.args.vocab.convert_tokens_to_ids(
            peptide[: self.args.seq_len]
        )
        ids = end_padding(ids, self.args.seq_len, self.args.vocab.pad_i)
        label = self.df["label"].iloc[item]
        standardized_label = self.args.scalers.transform([[label]])
        dict = {
            "token_ids": torch.tensor(ids),
            "standardized_label": torch.tensor(standardized_label[0][0]),
            "indx": self.df.index[item],
        }
        if self.args.mode == "supervised":
            if self.args.type =="pool":
                dict["features"]=torch.tensor(self.df.features.iloc[item])
        return dict

import numpy as np
def custom_collate(data):

    # Use the default collate function for everything except the task, this becomes a list of strings
    coll_data = default_collate(
        [
            {k: v for k, v in d.items() if k not in ["indx"]}
            for d in data
        ]
    )

    if "indx" in data[0]:
        coll_data["indx"] = [d["indx"] for d in data]

    return coll_data



