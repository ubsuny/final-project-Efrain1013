import argparse
import numpy as np
from pathlib import Path

from debug import (
    set_debug,
    log_debug_message,
    log_process_step,
    log_program_header,
    log_section_header,
    log_structure_summary,
    log_blank_line,
)
from sw_eval import evaluate_state
from relax_sd import run_neb_relaxation
from io_utils import read_poscar, write_neb_dat, write_neb_energy_dat, write_neb_images
import sw_build
from band import verify_endpoint_compatibility, build_interpolated_images, NEBBand
from band_metrics import evaluate_band_images, rebuild_band_neb_data


def parse_args():
    parser = argparse.ArgumentParser(description="Classical NEB relaxation driver using POSCAR input.")
    parser.add_argument("--initial-poscar", default="POSCAR-init", help="input initial POSCAR/CONTCAR file [default: POSCAR-init]")
    parser.add_argument("--final-poscar", default="POSCAR-final", help="input final POSCAR/CONTCAR file [default: POSCAR-final]")
    parser.add_argument("--debug", action="store_true", help="also print run.log output to the terminal")
    parser.add_argument("--pbc", action="store_true", help="enable periodic boundary conditions")
    parser.add_argument("--neb-file", default="NEB_Data/neb.dat", help="name of the output file for NEB data")
    parser.add_argument("--write-images", action="store_true", help="write final NEB image POSCAR files")
    parser.add_argument("--images-dir", default="NEB_Images", help="directory used with --write-images")
    parser.add_argument("--write-energies", action="store_true", help="write pair/triplet energy components using the NEB file prefix")

    parser.add_argument("--dr-max", type=float, default=0.05, help="maximum allowed atomic displacement per step (A)")
    parser.add_argument("--force-tol", type=float, default=5e-2, help="force convergence tolerance (eV/A)")
    parser.add_argument("--alpha", type=float, default=0.01, help="overdamped NEB step scale (A^2/eV)")

    parser.add_argument("--nimages", default=5, type=int, help="total number of images including endpoints")
    parser.add_argument("--max-steps", type=int, default=1000, help="maximum number of NEB relaxation steps")
    parser.add_argument("--spring-k", default=10.0, type=float, help="spring constant connecting neighboring images in NEB")
    return parser.parse_args()


def main():
    args = parse_args()
    no_pbc = not args.pbc

    set_debug(debug=args.debug)

    log_program_header("CLASSICAL NEB RELAXATION DRIVER")
    log_section_header("Run Settings")
    for key, units in [
        ("initial_poscar", None),
        ("final_poscar", None),
        ("debug", None),
        ("pbc", None),
        ("neb_file", None),
        ("write_images", None),
        ("images_dir", None),
        ("write_energies", None),
        ("dr_max", "A"),
        ("force_tol", "eV/A"),
        ("alpha", "A^2/eV"),
        ("nimages", None),
        ("max_steps", None),
        ("spring_k", None),
    ]:
        log_debug_message(key, getattr(args, key), units=units)

    log_section_header("Structure Input")
    log_process_step("Reading initial POSCAR structure")
    init_state = read_poscar(args.initial_poscar)
    log_debug_message("source_file", init_state.metadata.get("source_file"))
    log_debug_message("poscar_comment", init_state.metadata.get("poscar_comment"))
    log_debug_message("elements", init_state.metadata.get("elements"))
    log_debug_message("counts", init_state.metadata.get("counts"))
    log_debug_message("coord_type_input", init_state.metadata.get("coord_type_input"))
    log_debug_message("selective_dynamics", init_state.metadata.get("selective_dynamics"))
    log_structure_summary(init_state, label="Initial Structure (Fixed)")
    log_blank_line()

    log_process_step("Reading final POSCAR structure")
    final_state = read_poscar(args.final_poscar)
    log_debug_message("source_file", final_state.metadata.get("source_file"))
    log_debug_message("poscar_comment", final_state.metadata.get("poscar_comment"))
    log_debug_message("elements", final_state.metadata.get("elements"))
    log_debug_message("counts", final_state.metadata.get("counts"))
    log_debug_message("coord_type_input", final_state.metadata.get("coord_type_input"))
    log_debug_message("selective_dynamics", final_state.metadata.get("selective_dynamics"))
    log_structure_summary(final_state, label="Final Structure (Fixed)")

    log_process_step("Checking endpoint compatibility for NEB")
    verify_endpoint_compatibility(init_state, final_state)

    log_section_header("Potential Setup")
    log_process_step("Building SW potential parameters")
    pot_params = sw_build.build_sw_multi_params()
    log_debug_message("pot_params", pot_params)
    log_debug_message("init_state.type_ids", init_state.type_ids)
    log_debug_message("pair keys", list(pot_params.pair_params.keys()))
    log_debug_message("triplet keys", list(pot_params.triplet_params.keys()))

    log_section_header("Initial POSCAR Evaluation")
    log_process_step("Evaluating initial structure")
    evaluate_state(state=init_state, pot_params=pot_params, no_pbc=no_pbc)
    log_debug_message("initial_energy", init_state.energy, units="eV")
    initial_force_norms = np.linalg.norm(init_state.forces, axis=1)
    log_debug_message("initial_max_force", float(np.max(initial_force_norms)), units="eV/A")

    log_section_header("Final POSCAR Evaluation")
    log_process_step("Evaluating final structure")
    evaluate_state(state=final_state, pot_params=pot_params, no_pbc=no_pbc)
    log_debug_message("final_energy", final_state.energy, units="eV")
    final_force_norms = np.linalg.norm(final_state.forces, axis=1)
    log_debug_message("final_max_force", float(np.max(final_force_norms)), units="eV/A")

    log_section_header("Band Construction")
    log_process_step("Building interpolated images")
    images = build_interpolated_images(
        initial_state=init_state,
        final_state=final_state,
        nimages=args.nimages,
        pbc=not no_pbc,
    )

    log_process_step("Creating NEB band")
    band = NEBBand(
        images=images,
        spring_k=args.spring_k,
        force_tol=args.force_tol,
        max_steps=args.max_steps,
        dr_max=args.dr_max,
        alpha=args.alpha,
        pbc=not no_pbc,
    )
    log_debug_message("band_nimages", band.nimages)
    log_debug_message("band_interior_indices", list(band.interior_indices()))
    log_debug_message("band_endpoint_indices", band.endpoint_indices())

    log_section_header("Initial Band Evaluation")
    log_process_step("Evaluating all images in the initial band")
    evaluate_band_images(
        band=band,
        evaluate_fn=evaluate_state,
        pot_params=pot_params,
        no_pbc=no_pbc,
    )
    rebuild_band_neb_data(band)
    log_debug_message("initial_band_energies", band.energies)
    log_debug_message("initial_raw_forces_shape", band.raw_forces.shape)
    log_debug_message("initial_emax", band.emax, units="eV")
    log_debug_message("initial_imax", band.imax)
    log_debug_message("initial_fmax", band.fmax, units="eV/A")

    log_section_header("NEB Relaxation")
    log_process_step("Running damped NEB")
    band = run_neb_relaxation(
        band=band,
        evaluate_fn=evaluate_state,
        pot_params=pot_params,
        no_pbc=no_pbc,
        initial_evaluated=True,
    )

    written_files = []
    written_dirs = []
    write_neb_dat(band, args.neb_file)
    written_files.append(str(args.neb_file))
    if args.write_energies:
        path = Path(args.neb_file)
        prefix = path.with_suffix("") if path.suffix else path
        pair_file = prefix.parent / f"{prefix.name}_pairE.dat"
        triplet_file = prefix.parent / f"{prefix.name}_tripE.dat"
        pair_energies = np.array([img.metadata["components"]["pair"] for img in band.images])
        triplet_energies = np.array([img.metadata["components"]["triplet"] for img in band.images])
        write_neb_energy_dat(band, pair_energies, pair_file)
        write_neb_energy_dat(band, triplet_energies, triplet_file)
        written_files.extend([str(pair_file), str(triplet_file)])
    if args.write_images:
        write_neb_images(band, args.images_dir)
        written_dirs.append(str(args.images_dir))

    log_section_header("NEB Optimization Summary")
    log_debug_message("termination_reason", band.termination_reason)
    log_debug_message("converged", band.converged)
    log_debug_message("final_fmax", band.fmax, units="eV/A")
    log_debug_message("final_fperp_max", band.fperp_max, units="eV/A")
    log_debug_message("steps_taken", band.step + 1)
    log_debug_message("written_files")
    for filename in written_files:
        log_debug_message("output_file", filename)
    if written_dirs:
        log_debug_message("written_directories")
        for dirname in written_dirs:
            log_debug_message("output_directory", dirname)


if __name__ == "__main__":
    main()
