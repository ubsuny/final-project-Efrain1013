import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import fortran_backend

# store the result of one SW evaluation
@dataclass
class EvaluationResult:

    energy: float                               # eV
    forces: Optional[np.ndarray] = None         # shape (N,3), eV/Å

    # keep optional energy breakdowns here
    components: dict = field(default_factory=dict)
    # store backend bookkeeping here
    metadata: dict = field(default_factory=dict)


# ===============================================================================================
# FORTRAN FORTRAN FORTRAN FORTRAN FORTRAN FORTRAN FORTRAN FORTRAN FORTRAN FORTRAN FORTRAN FORTRAN 
# ===============================================================================================

# python uses 0-based indexing
PAIR_PARAM_EPSILON = 0
PAIR_PARAM_SIGMA   = 1
PAIR_PARAM_A       = 2
PAIR_PARAM_B       = 3
PAIR_PARAM_P       = 4
PAIR_PARAM_Q       = 5
PAIR_PARAM_SIGMA_P = 6
PAIR_PARAM_SIGMA_Q = 7
N_PAIR_PARAMS      = 8

TRIPLET_PARAM_EPSILON    = 0
TRIPLET_PARAM_LAMBDA     = 1
TRIPLET_PARAM_GAMMA_IJ   = 2
TRIPLET_PARAM_GAMMA_IK   = 3
TRIPLET_PARAM_SIGMA_IJ   = 4
TRIPLET_PARAM_SIGMA_IK   = 5
TRIPLET_PARAM_COS_THETA0 = 6
N_TRIPLET_PARAMS         = 7

# bundle the dense tables passed down to fortran
@dataclass
class SWfortranTables:
    zmax: int
    pair_exists: np.ndarray
    pair_rcut: np.ndarray
    pair_params: np.ndarray
    triplet_exists: np.ndarray
    triplet_rcut_ij: np.ndarray
    triplet_rcut_ik: np.ndarray
    triplet_params: np.ndarray

# find the largest atomic number needed by the dense Fortran tables
def infer_max_atomic_number(params, required_type_ids=None):
    z_values = set()

    # collect atomic numbers that appear in the SW parameter tables
    for ti, tj in params.pair_params.keys():
        z_values.add(int(ti))
        z_values.add(int(tj))

    for tc, tj, tk in params.triplet_params.keys():
        z_values.add(int(tc))
        z_values.add(int(tj))
        z_values.add(int(tk))

    # collect all unique z_values
    if required_type_ids is not None:
        for z in required_type_ids:
            z_values.add(int(z))
    # ensure we have z from our tables
    if not z_values:
        raise ValueError("Could not infer zmax from SW parameter tables.")

    return max(z_values)

# build dense lookup tables for the fortran evaluator (atomic number based)
def compile_sw_fortran_tables(params, required_type_ids=None):
    # get the highest atomic number
    zmax = infer_max_atomic_number(params, required_type_ids=required_type_ids)

    # arrays to ensure that pair and triplet parameters exist
    pair_exists = np.zeros((zmax + 1, zmax + 1), dtype=np.int32, order="F")
    pair_rcut   = np.zeros((zmax + 1, zmax + 1), dtype=np.float64, order="F")
    pair_params = np.zeros((zmax + 1, zmax + 1, N_PAIR_PARAMS), dtype=np.float64, order="F")

    triplet_exists  = np.zeros((zmax + 1, zmax + 1, zmax + 1), dtype=np.int32, order="F")
    triplet_rcut_ij = np.zeros((zmax + 1, zmax + 1, zmax + 1), dtype=np.float64, order="F")
    triplet_rcut_ik = np.zeros((zmax + 1, zmax + 1, zmax + 1), dtype=np.float64, order="F")
    triplet_params  = np.zeros((zmax + 1, zmax + 1, zmax + 1, N_TRIPLET_PARAMS), dtype=np.float64, order="F")

    # iterate over the pairs and collect parameters
    for (ti, tj), pair in params.pair_params.items():
        ti = int(ti)
        tj = int(tj)

        sigma_p = float(pair.sigma ** pair.p)
        sigma_q = float(pair.sigma ** pair.q)

        # store both orderings so fortran can index directly
        for a, b in ((ti, tj), (tj, ti)):
            pair_exists[a, b] = 1
            pair_rcut[a, b] = float(pair.r_cut)

            pair_params[a, b, PAIR_PARAM_EPSILON] = float(pair.epsilon)
            pair_params[a, b, PAIR_PARAM_SIGMA]   = float(pair.sigma)
            pair_params[a, b, PAIR_PARAM_A]       = float(pair.A)
            pair_params[a, b, PAIR_PARAM_B]       = float(pair.B)
            pair_params[a, b, PAIR_PARAM_P]       = float(pair.p)
            pair_params[a, b, PAIR_PARAM_Q]       = float(pair.q)
            pair_params[a, b, PAIR_PARAM_SIGMA_P] = sigma_p
            pair_params[a, b, PAIR_PARAM_SIGMA_Q] = sigma_q

    # iterate over triplets and collect parameters
    for (tc, tj, tk), trip in params.triplet_params.items():
        tc = int(tc)
        tj = int(tj)
        tk = int(tk)

        # Store the triplet exactly as the Fortran kernel indexes it: (center, j, k).
        triplet_exists[tc, tj, tk] = 1
        triplet_rcut_ij[tc, tj, tk] = float(trip.r_cut_ij)
        triplet_rcut_ik[tc, tj, tk] = float(trip.r_cut_ik)

        triplet_params[tc, tj, tk, TRIPLET_PARAM_EPSILON]    = float(trip.epsilon)
        triplet_params[tc, tj, tk, TRIPLET_PARAM_LAMBDA]     = float(trip.lam)
        triplet_params[tc, tj, tk, TRIPLET_PARAM_GAMMA_IJ]   = float(trip.gamma_ij)
        triplet_params[tc, tj, tk, TRIPLET_PARAM_GAMMA_IK]   = float(trip.gamma_ik)
        triplet_params[tc, tj, tk, TRIPLET_PARAM_SIGMA_IJ]   = float(trip.sigma_ij)
        triplet_params[tc, tj, tk, TRIPLET_PARAM_SIGMA_IK]   = float(trip.sigma_ik)
        triplet_params[tc, tj, tk, TRIPLET_PARAM_COS_THETA0] = float(trip.cos_theta0)

        # Also store (center, k, j) so the unique neighbor-pair loop can find
        # the same physical angle regardless of which neighbor appears first.
        triplet_exists[tc, tk, tj] = 1
        triplet_rcut_ij[tc, tk, tj] = float(trip.r_cut_ik)
        triplet_rcut_ik[tc, tk, tj] = float(trip.r_cut_ij)

        triplet_params[tc, tk, tj, TRIPLET_PARAM_EPSILON]    = float(trip.epsilon)
        triplet_params[tc, tk, tj, TRIPLET_PARAM_LAMBDA]     = float(trip.lam)
        triplet_params[tc, tk, tj, TRIPLET_PARAM_GAMMA_IJ]   = float(trip.gamma_ik)
        triplet_params[tc, tk, tj, TRIPLET_PARAM_GAMMA_IK]   = float(trip.gamma_ij)
        triplet_params[tc, tk, tj, TRIPLET_PARAM_SIGMA_IJ]   = float(trip.sigma_ik)
        triplet_params[tc, tk, tj, TRIPLET_PARAM_SIGMA_IK]   = float(trip.sigma_ij)
        triplet_params[tc, tk, tj, TRIPLET_PARAM_COS_THETA0] = float(trip.cos_theta0)

    if required_type_ids is not None:
        required = sorted(set(int(z) for z in required_type_ids))

        # ensure we successfully get our pair values
        for ti in required:
            for tj in required:
                if pair_exists[ti, tj] == 0:
                    raise ValueError(f"Missing fortran pair table entry for ({ti}, {tj})")
        # ensure we successfully get our triplet values
        for tc in required:
            for tj in required:
                for tk in required:
                    if triplet_exists[tc, tj, tk] == 0:
                        raise ValueError(f"Missing fortran triplet table entry for ({tc}, {tj}, {tk})")

    return SWfortranTables(
        zmax=zmax,
        pair_exists=pair_exists,
        pair_rcut=pair_rcut,
        pair_params=pair_params,
        triplet_exists=triplet_exists,
        triplet_rcut_ij=triplet_rcut_ij,
        triplet_rcut_ik=triplet_rcut_ik,
        triplet_params=triplet_params,
    )


# make a Fortran float array for the compiled kernel
def as_fortran_float64(arr):
    return np.asfortranarray(arr, dtype=np.float64)

# make a Fortran integer array for the compiled kernel
def as_fortran_int32(arr):
    return np.asfortranarray(arr, dtype=np.int32)

# evaluate the state with the fortran kernel
def evaluate_sw_fortran(state, params, no_pbc=False):
    # gather the state arrays in the layout expected by fortran
    positions = as_fortran_float64(state.positions_cart)
    lattice = as_fortran_float64(state.lattice)
    inv_lattice = as_fortran_float64(np.linalg.inv(lattice))
    type_ids = as_fortran_int32(state.type_ids)

    # check that the basic array shapes are what the fortran expects
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
    if lattice.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3), got {lattice.shape}")
    if inv_lattice.shape != (3, 3):
        raise ValueError(f"inv_lattice must have shape (3, 3), got {inv_lattice.shape}")
    if type_ids.ndim != 1 or type_ids.shape[0] != positions.shape[0]:
        raise ValueError(f"type_ids must have shape (N,), got {type_ids.shape} for N={positions.shape[0]}")

    # build dense pair and triplet tables for the species present in this structure
    tables = compile_sw_fortran_tables(params, required_type_ids=type_ids)

    pair_exists = as_fortran_int32(tables.pair_exists)
    pair_rcut = as_fortran_float64(tables.pair_rcut)
    pair_params = as_fortran_float64(tables.pair_params)

    triplet_exists = as_fortran_int32(tables.triplet_exists)
    triplet_rcut_ij = as_fortran_float64(tables.triplet_rcut_ij)
    triplet_rcut_ik = as_fortran_float64(tables.triplet_rcut_ik)
    triplet_params = as_fortran_float64(tables.triplet_params)

    # fortran kernel
    kernel = fortran_backend.fortran_backend.sw_eval_kernel

    kernel_result = kernel(
        positions,
        lattice,
        inv_lattice,
        type_ids,
        int(bool(no_pbc)),
        pair_exists,
        pair_rcut,
        pair_params,
        triplet_exists,
        triplet_rcut_ij,
        triplet_rcut_ik,
        triplet_params,
        natoms=int(positions.shape[0]),
        zmax=int(tables.zmax),
    )

    if len(kernel_result) != 6:
        raise RuntimeError(
            "The loaded fortran_backend extension is stale. Rebuild it after the "
            "pair_energy/triplet_energy interface update."
        )

    energy, forces, n_pairs, n_triplets, pair_energy, triplet_energy = kernel_result
    forces = np.asarray(forces, dtype=np.float64)

    return EvaluationResult(
        energy=float(energy),
        forces=forces,
        components={
            "pair": float(pair_energy),
            "triplet": float(triplet_energy),
        },
        metadata={
            "backend": "fortran",
            "n_pairs": int(n_pairs),
            "n_triplets": int(n_triplets),
            "pair_energy": float(pair_energy),
            "triplet_energy": float(triplet_energy),
            "zmax": int(tables.zmax),
        },
    )

# evaluate the current state and write the results back onto the state object.
def evaluate_state(state, pot_params, no_pbc=False):

    result = evaluate_sw_fortran(state=state, params=pot_params, no_pbc=no_pbc)

    # copy the evaluated quantities back onto the state
    state.energy = result.energy
    state.forces = result.forces

    if state.metadata is None:
        state.metadata = {}

    # split energy components and metadata
    state.metadata["components"] = result.components
    state.metadata["eval_metadata"] = result.metadata

    return result
