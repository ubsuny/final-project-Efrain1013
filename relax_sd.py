import numpy as np

from debug import calculate_table_widths, write_table_header, write_table_row
from band_metrics import evaluate_band
from structure import sync_cartesian_to_fractional, wrap_cartesian_positions


# run NEB relaxation for the band
def run_neb_relaxation(band, evaluate_fn, pot_params, no_pbc=False, initial_evaluated=False):
    band.termination_reason = None
    if not initial_evaluated:
        evaluate_band(band, evaluate_fn, pot_params, no_pbc=no_pbc)

    labels = ["step", "fperp_max", "frms", "alpha", "dr_max"]
    kinds = ["int", "float", "float", "float", "float"]
    widths = calculate_table_widths(labels=labels, kinds=kinds, precision=10)
    widths[3] = 13
    write_table_header(labels, widths=widths)

    alpha = band.alpha

    for step in range(band.max_steps):
        band.step = step
        row = [step, band.fperp_max, band.frms, alpha, band.dr_max]
        write_table_row(row, widths=widths, precision=10)

        if band.fperp_max < band.force_tol:
            break

        for i in band.interior_indices():
            # move only interior images
            dR = alpha * band.neb_forces[i]
            atom_norms = np.linalg.norm(dR, axis=1)
            max_atom_norm = float(np.max(atom_norms)) if atom_norms.size else 0.0
            if max_atom_norm > 1.0e-16 and max_atom_norm > band.dr_max:
                dR = dR * (band.dr_max / max_atom_norm)

            band.images[i].positions_cart = band.images[i].positions_cart + dR

            if band.pbc:
                band.images[i].positions_cart = wrap_cartesian_positions(band.images[i].positions_cart, band.images[i].lattice)

            sync_cartesian_to_fractional(band.images[i])
            band.images[i].energy = None
            band.images[i].forces = None
            band.images[i].stress = None
            band.images[i].pressure = None

        evaluate_band(band, evaluate_fn, pot_params, no_pbc=no_pbc)

    if band.termination_reason is None:
        if band.fperp_max < band.force_tol:
            band.converged = True
            band.termination_reason = "converged"
        else:
            band.termination_reason = "max_steps"

    return band
