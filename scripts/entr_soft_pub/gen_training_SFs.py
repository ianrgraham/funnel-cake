import argparse

from pathlib import Path

import gsd.hoomd
import numpy as np

from schmeud.dynamics import thermal
from schmeud import softness

import time
import gc

valid_input_formats = [".gsd"]
valid_output_formats = [".parquet"]

parser = argparse.ArgumentParser(description="Generate structure functions used to train softness")
parser.add_argument("ifile", type=str, help=f"Input file (allowed formats: {valid_input_formats}")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument('--chunks', help="Segment input files into N chunks", type=int)
parser.add_argument('--chunk-idx', help="If chunking, index to choose", type=int)
args = parser.parse_args()

ifile = Path(args.ifile)
ofile = Path(args.ofile)

chunks = args.chunks
chunk_idx = args.chunk_idx

print(f"Working file: {ifile}")

with gsd.hoomd.open(str(ifile), mode='rb') as traj:

    n_frames = len(traj)

    print("Calculating p_hop")
    phop = thermal.calc_phop(traj)

    idx_min = 0
    idx_max = len(phop)
    splits = np.linspace(idx_min, idx_max, chunks+1, dtype=int)
    sub_slice = slice(splits[chunk_idx], splits[chunk_idx+1])

    dyn_indices = softness.group_hard_soft_by_cutoffs(phop, hard_distance=400, sub_slice=sub_slice)

    print("Calculating features")

    print(len(dyn_indices))

    start = time.time()

    df = softness.calc_structure_functions_dataframe_rust(
        traj,
        dyn_indices=dyn_indices
    )

    print("generating training", time.time() - start)

    # phop = phop[sub_slice]
    sub_phop = []
    for idx, row in df[["frames", "ids"]].iterrows():
        sub_phop.append(phop[row.frames, row.ids])

    df["phop"] = sub_phop

df.to_parquet(ofile)
