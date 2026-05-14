import numpy as np

from band import collect_band_energy_force_arrays
from band_forces import build_improved_tangents, build_neb_forces

# evaluate each image in the band
def evaluate_band_images(band, evaluate_fn, pot_params, no_pbc=False):
    for img in band.images:
        evaluate_fn(state=img, pot_params=pot_params, no_pbc=no_pbc)


# compute band force metrics
def update_band_force_metrics(band):
    max_neb_force = 0.0
    max_perp_force = 0.0
    sumsq_perp = 0.0
    ncomp_perp = 0

    # loop over interior images only
    for i in band.interior_indices():

        # full neb force
        F_neb = band.neb_forces[i]

        # max norm of the neb force
        neb_atom_norms = np.linalg.norm(F_neb, axis=1)
        if neb_atom_norms.size > 0:
            max_neb_force = max(max_neb_force, float(np.max(neb_atom_norms)))

        # perpendicular component of true force
        F_perp = band.perpendicular_forces[i]

        # max atomic norm of the perp force
        perp_atom_norms = np.linalg.norm(F_perp, axis=1)
        if perp_atom_norms.size > 0:
            max_perp_force = max(max_perp_force, float(np.max(perp_atom_norms)))

        sumsq_perp += float(np.sum(F_perp * F_perp))
        ncomp_perp += int(F_perp.size)

    band.fneb_max = max_neb_force
    band.fperp_max = max_perp_force
    band.frms = float(np.sqrt(sumsq_perp / ncomp_perp)) if ncomp_perp > 0 else 0.0
    # choose max perp force as the force convergence metric
    band.fmax = band.fperp_max


# rebuild all neb band data
def rebuild_band_neb_data(band):
    # collect energies, coordinates, and raw forces
    collect_band_energy_force_arrays(band)
    # build improved Henkelman tangents from the current geometry
    build_improved_tangents(band)
    # build neb forces
    build_neb_forces(band)
    # update force metrics
    update_band_force_metrics(band)


# full band evaluation
def evaluate_band(band, evaluate_fn, pot_params, no_pbc=False):
    evaluate_band_images(band=band, evaluate_fn=evaluate_fn, pot_params=pot_params,
                         no_pbc=no_pbc)
    rebuild_band_neb_data(band)
