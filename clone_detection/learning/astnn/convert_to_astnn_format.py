from paths import *
import pandas as pd
import pandas.io.sql as pdsql
import psycopg2
import os

def convert(pairs, codes):
    pairs.columns = [x if (x != "type") else "label" for x in pairs.columns]
    print(pairs["label"].value_counts())
    codes = codes[["id", "code"]]
    codes.set_index('id', inplace=True)
    print(codes.head())
    return pairs, codes

if __name__ == "__main__":
    folder_path = tool_path + "/datasets/artset"
    codes = pd.read_csv(folder_path + "/fragments.csv")
    pairs = pd.read_csv(folder_path + "/pairs.csv")

    n_pairs, n_funcs = convert(pairs, codes)
    n_pairs.to_pickle(folder_path + "/bcb_pair_ids.pkl")
    n_funcs.to_csv(folder_path + "/bcb_funcs_all.tsv", sep="\t", header=None)


