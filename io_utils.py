import numpy as np
from pathlib import Path
from structure import (
    StructureState,
    sync_cartesian_to_fractional,
    sync_fractional_to_cartesian,
    image_displacement_cartesian,
)
from sw_params import ATOMIC_NUMBERS

# expands species into an array of species per atom
def expand_species_per_atom(elements, counts):
    species = []
    type_ids = []

    for elem, count in zip(elements, counts):
        z = ATOMIC_NUMBERS[elem]
        for atom in range(count):
            species.append(elem)
            type_ids.append(z)

    return species, np.array(type_ids, dtype=int)

# read in the specified POSCAR and return StructureState object
def read_poscar(poscar="POSCAR"):
    # see if POSCAR path exists
    if not Path(poscar).is_file():
        raise FileNotFoundError(f"Could not find requested file '{poscar}'. Did you set --initial/final-poscar ?")

    with open(poscar, "r", encoding="utf-8") as f:
        # keep non-empty lines only
        lines = [ln.strip() for ln in f if ln.strip()]

    # check that the POSCAR actually contains the coordinates
    if len(lines) < 8:
        raise ValueError(f"{poscar}: too few lines to be a valid POSCAR")

    i = 0

    # comment
    comment = lines[i]
    i += 1

    # scale factor
    scale = float(lines[i].split()[0])
    i += 1

    # lattice
    lattice_rows = []
    for row in range(3):
        parts = lines[i].split()
        if len(parts) < 3:
            raise ValueError(f"{poscar}: no good lattice line! --> {lines[i]!r}")
        lattice_rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        i += 1

    lattice = np.array(lattice_rows, dtype=float) * scale

    # species line
    elements = lines[i].split()
    if len(elements) == 0:
        raise ValueError(f"{poscar}: expected species line after lattice vectors")
    i += 1

    # counts line
    if i >= len(lines):
        raise ValueError(f"{poscar}: missing counts after species line")

    # ensure all the counts are integers
    count_tokens = lines[i].split()
    try:
        counts = [int(x) for x in count_tokens]
    except ValueError:
        raise ValueError(f"{poscar}: expected integer species counts after species line, got {lines[i]!r}")
    i += 1

    natoms = int(sum(counts))

    # optional selective dynamics
    selective_dynamics = False
    if lines[i].lower().startswith("s"):
        selective_dynamics = True
        i += 1

    # coordinate type
    coord_line = lines[i].lower()
    if coord_line.startswith("c") or coord_line.startswith("k"):
        coord_type = "Cartesian"
    else:
        coord_type = "Direct"
    i += 1

    # positions and optional selective flags
    positions_raw = np.zeros((natoms, 3), dtype=float)
    selective_flags = np.ones((natoms, 3), dtype=bool) if selective_dynamics else None

    # read in the raw positions for each atom
    for a in range(natoms):
        if i >= len(lines):
            raise ValueError(f"{poscar}: expected {natoms} position lines, got {a}")

        parts = lines[i].split()
        if len(parts) < 3:
            raise ValueError(f"{poscar}: no good position line! --> {lines[i]!r}")

        positions_raw[a, 0] = float(parts[0])
        positions_raw[a, 1] = float(parts[1])
        positions_raw[a, 2] = float(parts[2])

        # if selective dynamics are opted for the flags should be present
        if selective_dynamics:
            if len(parts) < 6:
                raise ValueError(f"{poscar}: selective dynamics enabled but missing T/F flags --> {lines[i]!r}")

            tf = [p.upper() for p in parts[3:6]]
            selective_flags[a, :] = [(x == "T") for x in tf]
        i += 1

    # build per-atom species/type info
    species, type_ids = expand_species_per_atom(elements, counts)

    # build state from whichever coordinate was given
    if coord_type == "Direct":
        state = StructureState(lattice=lattice, positions_cart=None, positions_frac=positions_raw.copy(), 
                               species=species, type_ids=type_ids)
        sync_fractional_to_cartesian(state)
    else:
        state = StructureState(lattice=lattice, positions_cart=positions_raw.copy(), positions_frac=None, 
                               species=species, type_ids=type_ids)
        sync_cartesian_to_fractional(state)

    # store input metadata
    state.metadata["source_file"] = str(poscar)
    state.metadata["poscar_comment"] = comment
    state.metadata["poscar_scale"] = scale
    state.metadata["elements"] = elements
    state.metadata["counts"] = counts
    state.metadata["coord_type_input"] = coord_type
    state.metadata["selective_dynamics"] = selective_dynamics
    state.metadata["selective_flags"] = selective_flags

    # make sure everything is valid
    state.validate_shapes()

    return state

# compute cumulative distance along the neb path
def compute_path_distances(band):
    n = band.nimages
    s = np.zeros(n)

    for i in range(1, n):
        dR = image_displacement_cartesian(band.images[i-1], band.images[i], pbc=band.pbc)
        ds = np.linalg.norm(dR)
        # configuration distance is a cumulative sum
        s[i] = s[i-1] + ds

    return s

# writes out neb data to file
def write_neb_dat(band, filename="neb.dat"):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    s = compute_path_distances(band)
    with open(filename, "w") as f:
        f.write("# image   distance   energy\n")
        for i in range(band.nimages):
            E = band.images[i].energy
            f.write(f"{i:4d}   {s[i]:12.6f}   {E:16.8f}\n")

# write energy data
def write_neb_energy_dat(band, energies, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    s = compute_path_distances(band)
    if len(energies) != band.nimages:
        raise ValueError("energy array length must match number of NEB images")

    with open(filename, "w") as f:
        f.write("# image   distance   energy\n")
        for i in range(band.nimages):
            f.write(f"{i:4d}   {s[i]:12.6f}   {energies[i]:16.8f}\n")


# write a StructureState in a compact POSCAR-like Cartesian format
def write_poscar(state, filename, comment="Generated POSCAR"):
    sync_cartesian_to_fractional(state)
    selective = state.metadata.get("selective_dynamics", False)
    flags = state.metadata.get("selective_flags", None)

    elements = []
    counts = []
    for elem in state.species:
        if elements and elem == elements[-1]:
            counts[-1] += 1
        else:
            elements.append(elem)
            counts.append(1)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{comment}\n")
        f.write("1.0\n")
        for row in state.lattice:
            f.write(f"{row[0]:22.16f} {row[1]:22.16f} {row[2]:22.16f}\n")
        f.write(" ".join(elements) + "\n")
        f.write(" ".join(str(c) for c in counts) + "\n")
        if selective:
            f.write("Selective Dynamics\n")
        f.write("Cartesian\n")
        for a, pos in enumerate(state.positions_cart):
            f.write(f"{pos[0]:22.16f} {pos[1]:22.16f} {pos[2]:22.16f}")
            if selective:
                atom_flags = flags[a] if flags is not None else np.ones(3, dtype=bool)
                f.write("  " + " ".join("T" if x else "F" for x in atom_flags))
            f.write("\n")


# write all final NEB image geometries for angle analysis
def write_neb_images(band, dirname="NEB_Images"):
    outdir = Path(dirname)
    outdir.mkdir(parents=True, exist_ok=True)
    for old_file in outdir.glob("POSCAR_image_*.vasp"):
        old_file.unlink()
    for i, img in enumerate(band.images):
        filename = outdir / f"POSCAR_image_{i:03d}.vasp"
        write_poscar(img, filename, comment=f"NEB image {i:03d}")
