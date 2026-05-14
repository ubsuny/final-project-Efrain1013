from itertools import combinations_with_replacement

import numpy as np

from sw_params import (
    ATOMIC_NUMBERS,
    PAIR_DB,
    TRIPLET_DB,
    SWMultiParams,
    SWPairParams,
    SWTripletParams,
)

# pair interactions are symmetric so store them in sorted order
def canonical_pair_key(ti, tj):
    return (ti, tj) if ti <= tj else (tj, ti)

# keep the center atom first but sort the two neighbors
def canonical_triplet_key(center, tj, tk):
    return (center, tj, tk) if tj <= tk else (center, tk, tj)

# list the atomic numbers that appear in a structure or type-id array
def present_atomic_types(state=None, type_ids=None):
    if state is not None:
        return sorted(set(int(x) for x in np.asarray(state.type_ids).tolist()))
    if type_ids is not None:
        return sorted(set(int(x) for x in type_ids))
    return []


# verify that every type actually present has all required pair/triplet entries
def validate_compiled_sw_params(params, state=None, type_ids=None):
    present = present_atomic_types(state=state, type_ids=type_ids)

    missing_pairs = []
    missing_triplets = []

    for ti, tj in combinations_with_replacement(present, 2):
        key = canonical_pair_key(ti, tj)
        if key not in params.pair_params:
            missing_pairs.append(key)

    for center in present:
        for tj, tk in combinations_with_replacement(present, 2):
            key = canonical_triplet_key(center, tj, tk)
            if key not in params.triplet_params:
                missing_triplets.append(key)

    if missing_pairs or missing_triplets:
        msg = "\nMissing SW parameters:\n"

        if missing_pairs:
            msg += "Pairs:\n"
            for key in missing_pairs:
                msg += f"  {key}\n"

        if missing_triplets:
            msg += "Triplets:\n"
            for key in missing_triplets:
                msg += f"  {key}\n"

        raise ValueError(msg)


def validate_species_map(species_map):
    if not isinstance(species_map, dict):
        raise ValueError("species_map must be a dictionary")

    seen_ids = set()
    for name, idx in species_map.items():
        if not isinstance(name, str):
            raise ValueError("species names must be strings")
        if not isinstance(idx, int):
            raise ValueError("species ids must be integers")
        if idx in seen_ids:
            raise ValueError(f"Duplicate species id detected: {idx}")
        seen_ids.add(idx)


def validate_pair_db(pair_db):
    for key, rec in pair_db.items():
        if len(key) != 2:
            raise ValueError(f"Invalid pair key {key}")
        if rec.r_cut <= 0.0:
            raise ValueError(f"Pair cutoff must be positive for {key}")


def validate_triplet_db(triplet_db):
    for key, rec in triplet_db.items():
        if len(key) != 3:
            raise ValueError(f"Invalid triplet key {key}")
        if rec.r_cut_ij <= 0.0 or rec.r_cut_ik <= 0.0:
            raise ValueError(f"Triplet cutoffs must be positive for {key}")



# compile tables into evaluator ready structures
def compile_pair_db(species_map, pair_db):
    pair_params = {}

    for (sa, sb), rec in pair_db.items():
        ta = species_map[sa]
        tb = species_map[sb]
        key = canonical_pair_key(ta, tb)

        pair_params[key] = SWPairParams(
            epsilon=rec.epsilon,
            sigma=rec.sigma,
            A=rec.A,
            B=rec.B,
            p=rec.p,
            q=rec.q,
            r_cut=rec.r_cut,
        )

    return pair_params


def compile_triplet_db(species_map, triplet_db):
    triplet_params = {}

    for (sc, sj, sk), rec in triplet_db.items():
        tc = species_map[sc]
        tj = species_map[sj]
        tk = species_map[sk]
        key = canonical_triplet_key(tc, tj, tk)

        triplet_params[key] = SWTripletParams(
            epsilon=rec.epsilon,
            lam=rec.lam,
            gamma_ij=rec.gamma_ij,
            gamma_ik=rec.gamma_ik,
            sigma_ij=rec.sigma_ij,
            sigma_ik=rec.sigma_ik,
            r_cut_ij=rec.r_cut_ij,
            r_cut_ik=rec.r_cut_ik,
            cos_theta0=rec.cos_theta0,
        )

    return triplet_params



def build_sw_multi_params_from_tables(species_map, pair_db, triplet_db, required_type_ids=None):
    validate_species_map(species_map)
    validate_pair_db(pair_db)
    validate_triplet_db(triplet_db)

    pair_params = compile_pair_db(species_map, pair_db)
    triplet_params = compile_triplet_db(species_map, triplet_db)

    params = SWMultiParams(pair_params=pair_params, triplet_params=triplet_params)

    if required_type_ids is not None:
        validate_compiled_sw_params(params, type_ids=required_type_ids)

    return params


# default builder using the tables defined in sw_params.py
def build_sw_multi_params():
    return build_sw_multi_params_from_tables( species_map=ATOMIC_NUMBERS, pair_db=PAIR_DB, triplet_db=TRIPLET_DB, 
                                             required_type_ids=sorted(ATOMIC_NUMBERS.values()))
