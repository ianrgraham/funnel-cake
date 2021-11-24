# TODO will need to change title here
import sys
import glob
import pickle
import os
import argparse

from pathlib import Path

import gsd.hoomd
import pandas as pd
import numpy as np

from schmeud import statics
from schmeud.utils import tail, parse_flag_from_string

valid_input_formats = [".gsd"]
valid_output_formats = [".pkl"]

parser = argparse.ArgumentParser(description="Construct excess entropy from strata of softness")
parser.add_argument("itraj", type=str, help=f"Input traj file (allowed formats: {valid_input_formats}")
parser.add_argument("isoft", type=str, help=f"Softness data file")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
args = parser.parse_args()

itraj = args.itraj
isoft = args.isoft
ofile = args.ofile

print(f"Working file: {isoft}")

# temp = parse_flag_from_string(isoft, "temp-")
# seed = parse_flag_from_string(isoft, "seed-", end=".")

# print(f"Temp: {temp}\nSeed: {seed}")
print(f"Original file: {itraj}")

traj = gsd.hoomd.open(itraj)
cuts = np.linspace(-1.0, 1.0, 41)
df = pd.read_parquet(isoft)
rdf_parts = statics.build_rdf_by_softness_for_traj(traj, df, cuts, bins=1000)
r = np.linspace(0.0, 5.0, 1001)

data = {"rad": r, "rdf_dict": rdf_parts, "soft_cuts": cuts}

print(f"Output file: {ofile}")

with open(ofile, 'wb') as f:
    pickle.dump(rdf_parts, f)