import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sw_params import PAIR_DB

xoffset = 1.8

# evaluate the analytic SW pair potential for separations
def sw_pair_energy(r, pair):
    r = np.asarray(r, dtype=float)
    energy = np.zeros_like(r)

    inside = (r > 0.0) & (r < pair.r_cut)
    sr_p = pair.sigma ** pair.p / r[inside] ** pair.p
    sr_q = pair.sigma ** pair.q / r[inside] ** pair.q
    cutoff = np.exp(pair.sigma / (r[inside] - pair.r_cut))

    energy[inside] = pair.epsilon * pair.A * (pair.B * sr_p - sr_q) * cutoff
    return energy


pair = PAIR_DB[("Si", "Si")]

data = np.loadtxt("./NEB_Data/neb_pair.dat", comments="#")
distance = data[:, 1]
energy_data = data[:, 2]

r_data = distance + xoffset
r_fine = np.linspace(r_data.min(), r_data.max(), 1000)
distance_fine = r_fine - xoffset
energy_sw = sw_pair_energy(r_fine, pair)

energy_sw_at_data = sw_pair_energy(r_data, pair)
rmse = np.sqrt(np.mean((energy_sw_at_data - energy_data) ** 2))
r_eq = distance_fine[np.argmin(energy_sw)] + xoffset
print(f"RMSE = {rmse:.5e} eV")
print(f"R_eq = {r_eq:.5f} Angstrom")

plt.figure(figsize=(5, 5))
plt.plot(distance_fine+xoffset, energy_sw, color="black", label="Expected SW", linewidth=2.5)
plt.scatter(distance+xoffset, energy_data, color="black", zorder=3, s=50, label="NEB")
plt.ylim(-2.5,1)
plt.axhline(0.0, alpha=0.4, color="black", linewidth=2, linestyle=":")
plt.axvline(r_eq, linewidth=3, linestyle=":", ymax=2.5/3.5,  color="red", label=rf"r$_{{eq}}$ = {r_eq:.2f} "+r"$\mathrm{\AA}$")
plt.xlabel(r"Distance ($\mathrm{\AA}$)")
plt.ylabel("Energy (eV)")

plt.title(f"SW Pair Potential Comparison")
plt.legend()
plt.tight_layout()
Path("./Plots").mkdir(parents=True, exist_ok=True)
plt.savefig("./Plots/sw_pair_overlay.png", dpi=200)
