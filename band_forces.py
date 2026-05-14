import numpy as np

from band import apply_selective_dynamics_mask
from structure import image_displacement_cartesian


# displacement between images that aren't fixed
def selective_dynamics_displacement(state_a, state_b, active_state, pbc=True):
    dR = image_displacement_cartesian(state_a, state_b, pbc=pbc)
    return apply_selective_dynamics_mask(dR, active_state)


# building improved tangents
def build_improved_tangents(band):
    nimages = band.nimages
    natoms = band.natoms

    tangents = np.zeros((nimages, natoms, 3), dtype=float)

    # avoid divide by zero during norm
    eps = 1e-12  

    # loop over interior images only
    for i in range(1, nimages - 1):
        R_im1 = band.images[i - 1]
        R_i   = band.images[i]
        R_ip1 = band.images[i + 1]

        # selected dynamic displacements
        dR_forward  = selective_dynamics_displacement(R_i,   R_ip1, active_state=R_i, pbc=band.pbc)
        dR_backward = selective_dynamics_displacement(R_im1, R_i,   active_state=R_i, pbc=band.pbc)

        # energy differences
        dE_forward  = R_ip1.energy - R_i.energy
        dE_backward = R_i.energy - R_im1.energy

        # uphill uses forward tangent
        if R_ip1.energy > R_i.energy > R_im1.energy:
            tau = dR_forward

        # downhill uses backward tangent
        elif R_ip1.energy < R_i.energy < R_im1.energy:
            tau = dR_backward

        # near extrema or mixed slopes use weighted
        else:
            abs_forward  = abs(dE_forward)
            abs_backward = abs(dE_backward)

            # energy changes
            dVmax = max(abs_forward, abs_backward)
            dVmin = min(abs_forward, abs_backward)

            # weight tangent toward largest slope
            if R_ip1.energy > R_im1.energy:
                tau = dVmax * dR_forward + dVmin * dR_backward
            else:
                tau = dVmin * dR_forward + dVmax * dR_backward

        norm = np.linalg.norm(tau)
        # store normalized tangent
        tangents[i] = tau / norm if norm > eps else 0.0

    band.tangents = tangents

# build the neb forces
def build_neb_forces(band):
    n = band.nimages
    neb_forces = [None] * n
    parallel_forces = [None] * n
    perpendicular_forces = [None] * n
    spring_forces = [None] * n

    # endpoints have zero force
    for i in (0, n - 1):
        z = np.zeros_like(band.raw_forces[i])
        neb_forces[i] = z
        parallel_forces[i] = z
        perpendicular_forces[i] = z
        spring_forces[i] = z

    # build forces for interior images
    for i in band.interior_indices():
        img = band.images[i]

        # remove fixed atom forces
        F = apply_selective_dynamics_mask(band.raw_forces[i], img)

        # tangent along the band
        tau = band.tangents[i]

        # decompose true force into parallel and perp.
        F_parallel = np.sum(F * tau) * tau
        F_perp = F - F_parallel

        # spring force only along tangent
        dR_forward = selective_dynamics_displacement(img, band.images[i + 1], active_state=img, pbc=band.pbc)
        dR_backward = selective_dynamics_displacement(band.images[i - 1], img, active_state=img, pbc=band.pbc)
        dist_forward = float(np.linalg.norm(dR_forward))
        dist_backward = float(np.linalg.norm(dR_backward))

        F_spring = band.spring_k * (dist_forward - dist_backward) * tau

        # neb force = perp true force + parallel spring force
        F_neb = F_perp + F_spring

        # store masked components
        parallel_forces[i] = apply_selective_dynamics_mask(F_parallel, img)
        perpendicular_forces[i] = apply_selective_dynamics_mask(F_perp, img)
        spring_forces[i] = apply_selective_dynamics_mask(F_spring, img)
        neb_forces[i] = apply_selective_dynamics_mask(F_neb, img)

    band.neb_forces = np.array(neb_forces, dtype=float)
    band.parallel_forces = np.array(parallel_forces, dtype=float)
    band.perpendicular_forces = np.array(perpendicular_forces, dtype=float)
    band.spring_forces = np.array(spring_forces, dtype=float)
