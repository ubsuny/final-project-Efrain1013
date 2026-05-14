from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from io_utils import read_poscar
from sw_params import PAIR_DB, TRIPLET_DB


ANGLE_ATOMS = (2, 1, 3)
REQ = 2.35255
SI_PAIR = PAIR_DB[("Si", "Si")]
GAMMA = np.degrees(np.arccos(1.0 - (0.5*(1.8*SI_PAIR.sigma)**2 / REQ**2)))

# calculates pair energy when r < r_c
def sw_pair_energy(r, pair):
    r = np.asarray(r, dtype=float)
    energy = np.zeros_like(r)

    inside = (r > 0.0) & (r < pair.r_cut)
    sr_p = pair.sigma ** pair.p / r[inside] ** pair.p
    sr_q = pair.sigma ** pair.q / r[inside] ** pair.q
    cutoff = np.exp(pair.sigma / (r[inside] - pair.r_cut))

    energy[inside] = pair.epsilon * pair.A * (pair.B * sr_p - sr_q) * cutoff
    return energy

# SW angular term for triplet
def sw_triplet_energy(rij_vec, rik_vec, triplet):
    rij = np.linalg.norm(rij_vec)
    rik = np.linalg.norm(rik_vec)
    if rij <= 0.0 or rik <= 0.0:
        return 0.0
    if rij >= triplet.r_cut_ij or rik >= triplet.r_cut_ik:
        return 0.0

    cos_theta = np.dot(rij_vec, rik_vec) / (rij * rik)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    expo = np.exp(triplet.gamma_ij * triplet.sigma_ij / (rij - triplet.r_cut_ij)
                    + triplet.gamma_ik * triplet.sigma_ik / (rik - triplet.r_cut_ik))
    return triplet.epsilon * triplet.lam * expo * (cos_theta - triplet.cos_theta0) ** 2

# calculate engle from positions
def angle_degrees(state, atom_i, atom_j, atom_k):
    p = state.positions_cart
    v1 = p[atom_i] - p[atom_j]
    v2 = p[atom_k] - p[atom_j]
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

# rebuild triplet energy
def sw_energy_components_three_atom(state):
    pair = SI_PAIR
    triplet = TRIPLET_DB[("Si", "Si", "Si")]
    p = state.positions_cart

    pair_energy = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            pair_energy += float(sw_pair_energy(np.linalg.norm(p[j] - p[i]), pair))

    triplet_energy = 0.0
    for center in range(3):
        neighbors = [idx for idx in range(3) if idx != center]
        rij_vec = p[neighbors[0]] - p[center]
        rik_vec = p[neighbors[1]] - p[center]
        triplet_energy += sw_triplet_energy(rij_vec, rik_vec, triplet)

    return pair_energy, triplet_energy


neb = np.loadtxt("./NEB_Data/neb_triplet.dat", comments="#")
if neb.ndim == 1:
    neb = neb.reshape(1, -1)

image_ids = neb[:, 0].astype(int)
energy_data = neb[:, 2]
atom_i, atom_j, atom_k = [idx - 1 for idx in ANGLE_ATOMS]

angles = []
energy_pair_images = []
energy_triplet_images = []
states = []
for image_id in image_ids:
    path = Path("./NEB_Triplet_Images") / f"POSCAR_image_{image_id:03d}.vasp"
    state = read_poscar(path)
    states.append(state)
    angles.append(angle_degrees(state, atom_i, atom_j, atom_k))
    pair_energy, triplet_energy = sw_energy_components_three_atom(state)
    energy_pair_images.append(pair_energy)
    energy_triplet_images.append(triplet_energy)

angles = np.array(angles)
energy_pair_images = np.array(energy_pair_images)
energy_triplet_images = np.array(energy_triplet_images)
energy_sw_images = energy_pair_images + energy_triplet_images

# first image is not a relaxed enpoing in triplet configuration
angles_plot = angles[1:]
energy_data_plot = energy_data[1:]
energy_sw_images_plot = energy_sw_images[1:]

# normalize to first image
energy_data_rel = energy_data_plot - energy_data_plot[0]
energy_sw_images_rel = energy_sw_images_plot - energy_sw_images_plot[0]


rmse = np.sqrt(np.mean((energy_sw_images_rel - energy_data_rel) ** 2))
theta0 = np.degrees(np.arccos(TRIPLET_DB[("Si", "Si", "Si")].cos_theta0))
print(f"RMSE = {rmse:.5e} eV")
print(f"Theta_0 = {theta0:.5f} degrees")

plt.figure(figsize=(5, 5))
order = np.argsort(angles_plot)
plt.plot(angles_plot[order], energy_sw_images_rel[order], color="black", label="Expected SW", linewidth=2.5)
plt.scatter(angles_plot, energy_data_rel, color="black", zorder=3, s=50, label="NEB")
plt.axhline(0.0, alpha=0.4, color="black", linewidth=2, linestyle="--")
plt.axvline(GAMMA, linewidth=3, linestyle=":", ymax=0.1/0.3,  color="red", label=rf"$\gamma$ = {GAMMA:.1f}$^\degree$")
plt.ylim(-0.05, 0.25)
plt.xlabel(r"Angle (degrees)")
plt.ylabel("Energy (eV)")
plt.title("SW Angular Potential Comparison")
plt.legend()
plt.tight_layout()
Path("./Plots").mkdir(parents=True, exist_ok=True)
plt.savefig("./Plots/sw_triplet_overlay.png", dpi=200)
