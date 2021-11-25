import pickle
import argparse
import scipy
import scipy.ndimage
import gsd.hoomd
import numpy as np
import pandas as pd

from pathlib import Path

from schmeud.dynamics import thermal
from schmeud import softness
import time
import gc

valid_input_formats = [".gsd"]
valid_output_formats = [".parquet"]

def local_s2(Xs):
        
    kinda_rdf = Xs[0::2] + Xs[1::2]
    r = np.linspace(0.1,5.0,50)
    r2 = (4*np.pi*np.power(r,2))
    almost_rdf = (kinda_rdf/r2)
    rdf_mean = np.mean(almost_rdf[20:])
    g = almost_rdf/rdf_mean
    s2 = -2*np.pi*np.nan_to_num((g*np.log(g) - g + 1)*r)*(r[1]-r[0])

    return np.sum(s2)

parser = argparse.ArgumentParser(description="Calculate softness and local excess entropy for A species of system")
parser.add_argument("ifile", type=str, help=f"Input file (allowed formats: {valid_input_formats}")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("--soft-pipe", type=str, help="Path to pipeline object")
parser.add_argument('--chunks', help="Segment input files into N chunks", type=int)
parser.add_argument('--chunk-idx', help="If chunking, index to choose", type=int)
parser.add_argument('--drop-xs', help="Stride of retained Xs", type=int, default=100)
args = parser.parse_args()

ifile = Path(args.ifile)
ofile = Path(args.ofile)

chunks = args.chunks
chunk_idx = args.chunk_idx

drop_xs = args.drop_xs

print(f"Working file: {ifile}")
print(f"Processing chunk: {chunk_idx} of {chunks}")

pipe_path_A = args.soft_pipe
with open(pipe_path_A, "rb") as f:
    pipe_dict = pickle.load(f)
    pipeA = pipe_dict["pipe"]

print(f"Pipeline extracted from {pipe_path_A}")

with gsd.hoomd.open(str(ifile), mode='rb') as traj:

    n_frames = len(traj)
    print(f"Frames within file: {n_frames}")

    print("Calculating p_hop")
    phop = thermal.calc_phop(traj)
    idx_min = 0
    idx_max = len(phop)
    splits = np.linspace(idx_min, idx_max, chunks+1, dtype=int)
    sub_slice = slice(splits[chunk_idx], splits[chunk_idx+1])

    phop_slice = phop[sub_slice].flatten()
    del phop

    gc.collect()

    print(f"Slice: {sub_slice}")

    start = time.time()

    print("Calculating features")
    df = softness.calc_structure_functions_dataframe_rust(
        traj,
        sub_slice=sub_slice
    )

    print("generating softness", time.time() - start)

    print("finding which intervals had rearrangements, and which did not")

    rearrang = phop_slice >= 0.2

    df["rearrang"] = rearrang
    df["phop"] = phop_slice

    # filter to A particles and transfrom Xs to softness
    print("Droping B type and calculating softness")
    df = df[df.labels == 0]
    df["softness"] = pipeA.decision_function(list(df.Xs.values))
    df["entropy"] = df.Xs.apply(local_s2)
    # df.drop("Xs", axis=1, inplace=True)
    df["Xs"][df["frames"] % 10 != 0] = None

print(f"Writing to parquet: {ofile}")
df.to_parquet(str(ofile))
