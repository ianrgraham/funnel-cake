from typing import DefaultDict
import freud
import numpy as np
import gsd.hoomd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.stats
import argparse
import pathlib

import warnings

from itertools import product

from schmeud.utils import parse_flag_from_string

mpl.rcParams.update({'font.size': 15})
mpl.rcParams.update({'figure.dpi': 200})
mpl.rcParams.update({'figure.figsize': (6,4)})

valid_input_formats = [".gsd"]
valid_output_formats = [".flag"]

parser = argparse.ArgumentParser(description="Train softness on A species of given file")
parser.add_argument("ifiles", type=str, nargs="+", help=f"Input files (allowed formats: {valid_input_formats}")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("--time-interval", type=float, nargs=2, help="Time interval ")
args = parser.parse_args()

ifiles = args.ifiles
ofile = pathlib.Path(args.ofile)

# plots to be made 
# 1) log-log MSD over time
# 2) linear MSD over time
# 3) log dr^2/dt over time
# 4) D vs. 1/T

def calculate_msd(file, skip=None):

    traj = gsd.hoomd.open(file, mode='rb')
    snap = traj[0]
    gsd_box = snap.configuration.box

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        box = freud.box.Box(*gsd_box)

    msd = freud.msd.MSD(mode='window')

    # properly wrap trajectory
    for enum_i, idx in enumerate(range(0,len(traj),skip)):
        if enum_i == 0:
            last_pos = traj[idx].particles.position
            pos = []
            pos.append(last_pos)
        else:
            snap = traj[idx]
            next_pos = snap.particles.position
            dx = box.wrap(next_pos - last_pos)

            pos.append(pos[-1] + dx)
            last_pos = next_pos

    pos = np.array(pos)
    computed_msd = msd.compute(pos).msd

    return computed_msd

def plot_msd(time, msds):
    pass

def plot_dr2_dt(time, msds):
    cmap = cm.jet
    norm = colors.Normalize(vmin=min_temp, vmax=max_temp)
    symbols = [".", "D", "x", "o"]
    short_time = time[:-1]
    for idx, (pair, (temp, msd)) in msds.items():
        plt.plot(short_time, msd, symbols[idx], label=pair, color=cmap(norm(temp)))

    plt.yscale('log')

def plot_D_vs_invT(time, msds):
    
    result = scipy.stats.linregress(x=time[400:-100], y=msd[400:-100])
    pass

N = 10_000
skip = 10
min_temp = 0.45
max_temp = 1.0

time = np.linspace(0, N, N//skip)

msds = DefaultDict(list)
for file in ifiles:
    system_tag = file.split("/")[-2]
    temp = float(parse_flag_from_string(file, "temp-"))
    msds[system_tag].append((temp, calculate_msd(file, skip=skip)))

ofile.touch()