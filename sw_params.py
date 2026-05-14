from dataclasses import dataclass


# pair parameters
@dataclass
class SWPairParams:
    epsilon: float
    sigma: float
    A: float
    B: float
    p: float
    q: float
    r_cut: float

# triplet parameters
@dataclass
class SWTripletParams:
    epsilon: float
    lam: float
    gamma_ij: float
    gamma_ik: float
    sigma_ij: float
    sigma_ik: float
    r_cut_ij: float
    r_cut_ik: float
    cos_theta0: float

# builds parameters with multiple species
@dataclass
class SWMultiParams:
    # compiled dictionaries keyed by integer type ids
    pair_params: dict
    triplet_params: dict

    def __repr__(self):
        return (
            "SWMultiParams("
            f"n_pair_types={len(self.pair_params)}, "
            f"n_triplet_types={len(self.triplet_params)})"
        )

# helpers to make it easire to read which parameters are specified later
@dataclass
class SWPairRecord:
    epsilon: float
    sigma: float
    A: float
    B: float
    p: float
    q: float
    r_cut: float

@dataclass
class SWTripletRecord:
    epsilon: float
    lam: float
    gamma_ij: float
    gamma_ik: float
    sigma_ij: float
    sigma_ik: float
    r_cut_ij: float
    r_cut_ik: float
    cos_theta0: float


# currently supported elements
ATOMIC_NUMBERS = {
    "Si": 14,
}

# original Stillinger-Weber Si parameters in eV and Angstrom.
PAIR_DB = {
    ("Si", "Si"): SWPairRecord(
        epsilon=2.1683,
        sigma=2.0951,
        A=7.049556277,
        B=0.6022245584,
        p=4.0,
        q=0.0,
        r_cut=3.77118,   # = 1.8 * 2.0951
    ),
}

# triplet term favors tetrahedral Si through cos(theta0) = -1/3.
TRIPLET_DB = {
    ("Si", "Si", "Si"): SWTripletRecord(
        epsilon=2.1683,
        lam=21.0,
        gamma_ij=1.20,
        gamma_ik=1.20,
        sigma_ij=2.0951,
        sigma_ik=2.0951,
        r_cut_ij=3.77118,
        r_cut_ik=3.77118,
        cos_theta0=-1.0 / 3.0,
    ),
}
