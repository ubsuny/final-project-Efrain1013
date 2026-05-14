import numpy as np

from structure import sync_cartesian_to_fractional, wrap_cartesian_positions, image_displacement_cartesian
from debug import calculate_table_widths, write_table_header, write_table_row


# container for neb band and its state
class NEBBand:
    def __init__(self, images, spring_k, force_tol, max_steps, dr_max, alpha, pbc=True):
        if len(images) < 3:
            raise ValueError("NEBBand requires at least 3 images (initial, at least one interior image, and final).")

        # band and structure information
        self.images = images
        self.nimages = len(images)
        self.natoms = images[0].natoms()
        self.lattice = images[0].lattice.copy()
        self.pbc = pbc

        # core neb controls
        self.spring_k = spring_k
        self.force_tol = force_tol
        self.max_steps = max_steps
        self.dr_max = dr_max
        self.alpha = alpha

        # per image/per band neb arrays
        self.energies = None
        self.raw_forces = None
        self.tangents = None
        self.parallel_forces = None
        self.perpendicular_forces = None
        self.spring_forces = None
        self.neb_forces = None
        
        # band metrics
        self.emax = None
        self.emin = None
        self.imax = None
        self.imin = None
        self.fmax = None
        self.frms = None

        # convergence
        self.fperp_max = None
        self.termination_reason = None
        self.converged = False

        self.step = 0
    
    # returns all interior indices
    def interior_indices(self):
        return range(1, self.nimages - 1)
    # returns endpoint indices
    def endpoint_indices(self):
        return (0, self.nimages - 1)


# verify that the initial and final endpoint structures can form a NEB band
def verify_endpoint_compatibility(state1, state2):
    natoms1 = state1.natoms()
    natoms2 = state2.natoms()
    if natoms1 != natoms2:
        raise ValueError(f"NEB endpoint mismatch: state1 has {natoms1} atoms but state2 has {natoms2} atoms")
    if state1.species != state2.species:
        raise ValueError("NEB endpoint mismatch: species ordering differs between the endpoints")
    if state1.type_ids.shape != state2.type_ids.shape:
        raise ValueError("NEB endpoint mismatch: type_ids have different shapes")
    if not np.array_equal(state1.type_ids, state2.type_ids):
        raise ValueError("NEB endpoint mismatch: type_ids differ between the two states")
    if state1.lattice.shape != state2.lattice.shape:
        raise ValueError("NEB endpoint mismatch: lattices have different shapes")
    if not np.allclose(state1.lattice, state2.lattice):
        raise ValueError("NEB requires the same lattice for endpoints")
    if state1.positions_cart.shape != state2.positions_cart.shape:
        raise ValueError("NEB endpoint mismatch: positions_cart have different shapes")
    if state1.positions_frac.shape != state2.positions_frac.shape:
        raise ValueError("NEB endpoint mismatch: positions_frac have different shapes")

# collect image energies and raw forces into band level arrays
def collect_band_energy_force_arrays(band):
    # loop over all images
    for i, img in enumerate(band.images):
        if img.energy is None:
            raise ValueError(f"Image {i} has no energy / did we evaluate the band first")
        if img.forces is None:
            raise ValueError(f"Image {i} has no forces / did we evaluate the band first")
    band.energies = np.array([img.energy for img in band.images], dtype=float)
    band.raw_forces = np.array([img.forces for img in band.images], dtype=float)

    band.emax = float(np.max(band.energies))
    band.emin = float(np.min(band.energies))
    band.imax = int(np.argmax(band.energies))
    band.imin = int(np.argmin(band.energies))

# build a linearly interpolated image list including fixed endpoints
def build_interpolated_images(initial_state, final_state, nimages, pbc=True):
    if nimages < 3:
        raise ValueError("nimages must be at least 3 (initial, at least one interior image, and final).")

    images = []
    dR = image_displacement_cartesian(initial_state, final_state, pbc=pbc)
    dR_norms = np.linalg.norm(dR, axis=1)
    max_dR_norm = float(np.max(dR_norms)) if dR_norms.size > 0 else 0.0

    # evenly space images along linear interpolation
    x_values = np.linspace(0.0, 1.0, nimages)
    dx_values = np.zeros(nimages, dtype=float)
    dx_values[1:] = x_values[1:] - x_values[:-1]

    # output table that describes the interpolated image information
    labels = ["Image", "x (%)", "dx", "dx * max|dR| (Angstrom)"]
    kinds = ["int", "float", "float", "float"]
    widths = calculate_table_widths(labels=labels, kinds=kinds, precision=10)
    widths[labels.index("x (%)")] = max(widths[labels.index("x (%)")], 15)
    write_table_header(labels, widths=widths)

    # loop over each image
    for i in range(nimages):
        x = float(x_values[i])
        dx = float(dx_values[i])
        write_table_row([i, 100.0 * x, dx, dx * max_dR_norm], widths=widths, precision=10)

        # fix endpoints and create intermediate image structures
        if i == 0:
            img = initial_state.copy()
        elif i == nimages - 1:
            img = final_state.copy()
        else:
            img = initial_state.copy()
            img.positions_cart = initial_state.positions_cart + x * dR
            if pbc:
                img.positions_cart = wrap_cartesian_positions(img.positions_cart, img.lattice)
            sync_cartesian_to_fractional(img)
            img.energy = None
            img.forces = None
            img.stress = None
            img.pressure = None

        images.append(img)

    if not np.allclose(images[0].positions_cart, initial_state.positions_cart):
        raise ValueError("Interpolated initial image does not match supplied initial state.")
    if not np.allclose(images[-1].positions_cart, final_state.positions_cart):
        raise ValueError("Interpolated final image does not match supplied final state.")

    return images

# apply selective dynamics to steps 
def apply_selective_dynamics_mask(arr, state):
    selective = state.metadata.get("selective_dynamics", False)
    flags = state.metadata.get("selective_flags", None)
    if not selective or flags is None:
        return arr
    return arr * flags.astype(float)
