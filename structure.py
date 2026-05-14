import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# convert fractional coordinates to cartesian coordinates
def fractional_to_cartesian(frac, lattice):
    frac = np.asarray(frac, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    return frac @ lattice


# convert cartesian coordinates to fractional coordinates
def cartesian_to_fractional(cart, lattice):
    cart = np.asarray(cart, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    inv_lat = np.linalg.inv(lattice)
    return cart @ inv_lat


# wrap fractional coordinates into [0,1)
def wrap_fractional_positions(frac):
    frac = np.asarray(frac, dtype=float)
    return frac % 1.0


# wrap cartesian positions to the unit cell by converting to fractional first
def wrap_cartesian_positions(cart, lattice):
    frac = cartesian_to_fractional(cart, lattice)
    frac_wrapped = wrap_fractional_positions(frac)
    return fractional_to_cartesian(frac_wrapped, lattice)


# find minimum-image cartesian displacements
def minimum_image_displacement(dr_cart, lattice=None, no_pbc=False):
    dr_cart = np.asarray(dr_cart, dtype=float)

    if no_pbc or lattice is None:
        return dr_cart

    lattice = np.asarray(lattice, dtype=float)
    inv_lat = np.linalg.inv(lattice)

    dr_frac = dr_cart @ inv_lat
    dr_frac -= np.round(dr_frac)
    return dr_frac @ lattice


# get the full cartesian displacement of atoms between two states
def image_displacement_cartesian(state_a, state_b, pbc=True):
    dr = state_b.positions_cart - state_a.positions_cart
    return minimum_image_displacement(dr, lattice=state_a.lattice, no_pbc=not pbc)


# StructureState stores one complete atomic configuration
@dataclass
class StructureState:

    lattice: np.ndarray                  # shape (3, 3), Angstrom
    positions_cart: np.ndarray           # shape (N, 3), Angstrom
    species: list                        # length N
    type_ids: np.ndarray                 # shape (N,), int

    positions_frac: Optional[np.ndarray] = None   # shape (N, 3), dimensionless

    energy: Optional[float] = None                # eV
    forces: Optional[np.ndarray] = None           # shape (N, 3), eV/Angstrom
    stress: Optional[np.ndarray] = None           # shape (3, 3), eV/Angstrom^3
    pressure: Optional[float] = None              # eV/Angstrom^3

    # metadata stores optional input flags, bookkeeping, and energy components
    metadata: dict = field(default_factory=dict)

    # returns number of atoms
    def natoms(self):
        return int(self.positions_cart.shape[0])

    # returns a deep copy so images can be moved independently
    def copy(self):
        return copy.deepcopy(self)

    # basic consistency checks
    def validate_shapes(self):
        if self.lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3,3), got {}".format(self.lattice.shape))

        if self.positions_cart.ndim != 2 or self.positions_cart.shape[1] != 3:
            raise ValueError("positions_cart must have shape (N,3), got {}".format(self.positions_cart.shape))

        n = self.positions_cart.shape[0]

        if len(self.species) != n:
            raise ValueError("species length ({}) does not match number of atoms ({})".format(len(self.species), n))

        if self.type_ids.shape != (n,):
            raise ValueError("type_ids must have shape ({},), got {}".format(n, self.type_ids.shape))

        if self.positions_frac is not None and self.positions_frac.shape != (n, 3):
            raise ValueError("positions_frac must have shape ({},3), got {}".format(n, self.positions_frac.shape))


# update fractional positions from cartesian positions
def sync_cartesian_to_fractional(state):
    state.positions_frac = cartesian_to_fractional(state.positions_cart, state.lattice)


# update cartesian positions from fractional positions
def sync_fractional_to_cartesian(state):
    if state.positions_frac is None:
        raise ValueError("state.positions_frac is None; cannot sync to cartesian.")

    state.positions_cart = fractional_to_cartesian(state.positions_frac, state.lattice)
