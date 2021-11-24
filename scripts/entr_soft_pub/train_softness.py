import pickle
import argparse

from pathlib import Path
import sys

import pandas as pd

from schmeud import softness

valid_input_formats = [".parquet"]
valid_output_formats = [".pkl"]

parser = argparse.ArgumentParser(description="Train softness on A species of given file")
parser.add_argument("ifiles", type=str, nargs="+", help=f"Input files (allowed formats: {valid_input_formats}")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("--sample", type=int)
parser.add_argument("--max-iter", type=int, default=100_000)
args = parser.parse_args()

ifiles = args.ifiles
ofile = Path(args.ofile)

sample = args.sample
max_iter = args.max_iter

dfs = []
for ifile in ifiles:
    dfs.append(pd.read_parquet(ifile))

df = pd.concat(dfs)
del(dfs)

df_a = df[df.labels == 0]

data_size = df_a.ys.value_counts()

pair = str(ofile).split('/')[-2]

print(pair, df_a.ys.value_counts().min())

if sample is not None:
    df_a = df_a.groupby('ys').apply(lambda x: x.sample(sample)).reset_index(drop=True)
    samples_used = sample
else:
    df_a = df_a.groupby('ys').apply(lambda x: x.sample(data_size.min())).reset_index(drop=True)
    samples_used = data_size.min()

pipe, acc = softness.train_hyperplane_pipeline(df_a.Xs, df_a.ys, max_iter=max_iter)
print(f"for {pair}")

with open(ofile, "wb") as f:
    pickle.dump({"pipe": pipe, "acc": acc, "data_size": data_size, "samples": samples_used}, f)