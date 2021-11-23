# import necessary files
import sys
import os
import glob
import argparse

from pathlib import Path

import gsd.hoomd

from dotenv import dotenv_values

CONFIG = dotenv_values()
tmp_path = CONFIG["SOFTRHEO_ROOT_DIR"]
assert(isinstance(tmp_path, str))
root_dir = Path(tmp_path)
tmp_path = CONFIG["SOFTRHEO_DATA_DIR"]
assert(isinstance(tmp_path, str))
data_dir = Path(tmp_path)
if root_dir not in sys.path:
    sys.path.insert(1, str(root_dir))

from softrheo.dynamics import thermal  # noqa
from softrheo.ml import softness  # noqa

parser = argparse.ArgumentParser(description="Build softness grouped g(r)")
parser.add_argument('-i', '--idx', help="File index of the dataset to grab", type=int, default=0)
parser.add_argument('-l', '--len', help="Check dataset size and escape", action='store_true')
cmdargs = parser.parse_args()

try:
    array_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
    file_path = f"{data_dir}/LJ_NVT/*N4096*T0.47*"
except Exception:
    array_id = 0
    file_path = f"{data_dir}/LJ_NVT/LJ_NVT_N512_phi1.2_T0.47_seed1.gsd"

file_idx = cmdargs.idx
file_paths = sorted(glob.glob(f"{data_dir}/LJ_NVT/*N4096*T0.47*"))

if cmdargs.len:
    print(f"Dataset files: {len(file_paths)}")
    sys.exit()

file_path = file_paths[file_idx]

print(f"Working file: {file_path}")

out_path = file_path.replace(
    "/LJ_NVT/",
    "/LJ_NVT_training_SFs_phop/"
    ).replace(
    ".gsd",
    "_training_SFs.parquet"
    )

os.makedirs('/'.join(out_path.split('/')[:-1]), exist_ok=True)

with gsd.hoomd.open(file_path, mode='rb') as traj:

    n_frames = len(traj)
    print(n_frames)

    print("Calculating p_hop")
    phop = thermal.calc_phop(traj)

    dyn_indices = softness.group_hard_soft_by_cutoffs(phop, hard_distance=400)

    print("Calculating features")
    df = softness.calc_structure_functions_dataframe(
        traj,
        dyn_indices=dyn_indices,
    )

    sub_phop = []
    for idx, row in df[["frames", "ids"]].iterrows():
        sub_phop.append(phop[row.frames, row.ids])

    df["phop"] = sub_phop

df.to_parquet(out_path)
