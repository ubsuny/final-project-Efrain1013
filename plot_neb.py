import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def parse_args():
    parser = argparse.ArgumentParser(description="Plot an NEB energy profile.")
    parser.add_argument("--neb-file", default="NEB_Data/neb.dat", help="NEB data file to plot")
    parser.add_argument("--out-file", default="./Plots/neb_plot.png", help="output image file")
    parser.add_argument("--title", default="NEB Energy Profile", help="plot title")
    return parser.parse_args()
args = parse_args()

data = np.loadtxt(args.neb_file)
s = data[:, 1]
energy = data[:, 2]
E = energy - energy[0]

# PCHIP linear interpolator
cs = PchipInterpolator(s, E)
sf = np.linspace(s.min(), s.max(), 400)

barrier = E.max() - E[0]
print(f"Barrier = {barrier:.5f} eV")

plt.figure(figsize=(5, 5))
plt.plot(sf, cs(sf), color="black", linewidth=2.5)
plt.scatter(s, E, color="black", zorder=3, s=50, label="NEB Images")
plt.axhline(0.0, alpha=0.4, color="black", linewidth=2, linestyle=":")
plt.xlabel(r"Displacement ($\mathrm{\AA}$)")
plt.ylabel("Energy (eV)")
plt.title(args.title)
plt.tight_layout()
plt.legend()
Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out_file, dpi=200)
