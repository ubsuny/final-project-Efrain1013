MODULE fortran_backend
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: sw_eval_kernel

    INTEGER, PARAMETER :: dp = KIND(1.0D0)

    ! pair parameters in pair_params(:, :, :)
    INTEGER, PARAMETER :: P_EPSILON = 1
    INTEGER, PARAMETER :: P_SIGMA   = 2
    INTEGER, PARAMETER :: P_A       = 3
    INTEGER, PARAMETER :: P_B       = 4
    INTEGER, PARAMETER :: P_P       = 5
    INTEGER, PARAMETER :: P_Q       = 6
    INTEGER, PARAMETER :: P_SIGMA_P = 7
    INTEGER, PARAMETER :: P_SIGMA_Q = 8

    ! triplet parameters in triplet_params(:, :, :, :)
    INTEGER, PARAMETER :: T_EPSILON   = 1
    INTEGER, PARAMETER :: T_LAMBDA    = 2
    INTEGER, PARAMETER :: T_GAMMA_IJ  = 3
    INTEGER, PARAMETER :: T_GAMMA_IK  = 4
    INTEGER, PARAMETER :: T_SIGMA_IJ  = 5
    INTEGER, PARAMETER :: T_SIGMA_IK  = 6
    INTEGER, PARAMETER :: T_COSTHETA0 = 7

CONTAINS

    ! main kernel
    SUBROUTINE sw_eval_kernel(natoms, zmax, positions, lattice, inv_lattice, type_ids, &
                              no_pbc, pair_exists, pair_rcut, &
                              pair_params, triplet_exists, triplet_rcut_ij, triplet_rcut_ik, &
                              triplet_params, energy, forces, n_pairs, n_triplets, &
                              pair_energy, triplet_energy)
        IMPLICIT NONE

        ! inputs
        INTEGER, INTENT(IN) :: natoms, zmax, no_pbc
        INTEGER, INTENT(IN) :: type_ids(natoms), pair_exists(0:zmax, 0:zmax), triplet_exists(0:zmax, 0:zmax, 0:zmax)
        DOUBLE PRECISION, INTENT(IN) :: positions(natoms, 3), lattice(3, 3), inv_lattice(3, 3)
        DOUBLE PRECISION, INTENT(IN) :: pair_rcut(0:zmax, 0:zmax), pair_params(0:zmax, 0:zmax, 8)
        DOUBLE PRECISION, INTENT(IN) :: triplet_rcut_ij(0:zmax, 0:zmax, 0:zmax), triplet_rcut_ik(0:zmax, 0:zmax, 0:zmax), &
                                        triplet_params(0:zmax, 0:zmax, 0:zmax, 7)     

        ! outputs
        DOUBLE PRECISION, INTENT(OUT) :: energy
        ! Component energies are returned for pair/triplet diagnostic plots.
        DOUBLE PRECISION, INTENT(OUT) :: pair_energy, triplet_energy
        DOUBLE PRECISION, INTENT(OUT) :: forces(natoms, 3)
        INTEGER, INTENT(OUT) :: n_pairs, n_triplets

        ! Locals
        INTEGER :: i, j, k
        INTEGER :: ti, tj, tk
        INTEGER :: nneigh, a, b

        DOUBLE PRECISION :: rij_vec(3), rik_vec(3)
        DOUBLE PRECISION :: rij, rik
        DOUBLE PRECISION :: rcut_ij, rcut_ik
        DOUBLE PRECISION :: v2, v3
        DOUBLE PRECISION :: dvdr
        DOUBLE PRECISION :: fij(3)
        DOUBLE PRECISION :: dh_drij(3), dh_drik(3)
        DOUBLE PRECISION :: volume
        DOUBLE PRECISION :: max_pair_rcut, max_triplet_rcut, rcut_build

        INTEGER, ALLOCATABLE :: neigh_idx(:)
        DOUBLE PRECISION, ALLOCATABLE :: neigh_vec(:, :)
        DOUBLE PRECISION, ALLOCATABLE :: neigh_dist(:)

        ! Initialize outputs
        energy = 0.0_dp
        pair_energy = 0.0_dp
        triplet_energy = 0.0_dp
        forces = 0.0_dp
        n_pairs = 0
        n_triplets = 0

        volume = cell_volume(lattice)

        ! Build a general cutoff for centered neighbor construction
        max_pair_rcut = MAXVAL(pair_rcut)

        max_triplet_rcut = 0.0_dp
        IF (MAXVAL(triplet_rcut_ij) > max_triplet_rcut) max_triplet_rcut = MAXVAL(triplet_rcut_ij)
        IF (MAXVAL(triplet_rcut_ik) > max_triplet_rcut) max_triplet_rcut = MAXVAL(triplet_rcut_ik)

        rcut_build = MAX(max_pair_rcut, max_triplet_rcut)


        ! Pair loop

        DO i = 1, natoms - 1
            ti = type_ids(i)

            DO j = i + 1, natoms
                tj = type_ids(j)

                IF (pair_exists(ti, tj) == 0) CYCLE

                CALL displacement_minimum_image(positions(i, :), positions(j, :), &
                                                lattice, inv_lattice, no_pbc, rij_vec)

                rij = norm3(rij_vec)
                rcut_ij = pair_rcut(ti, tj)

                IF (rij >= rcut_ij) CYCLE
                IF (rij <= 0.0_dp) CYCLE

                CALL sw_pair_energy_derivative(rij, pair_params(ti, tj, :), rcut_ij, v2, dvdr)

                energy = energy + v2
                pair_energy = pair_energy + v2
                n_pairs = n_pairs + 1

                fij(1) = (dvdr / rij) * rij_vec(1)
                fij(2) = (dvdr / rij) * rij_vec(2)
                fij(3) = (dvdr / rij) * rij_vec(3)

                forces(i,1) = forces(i,1) + fij(1)
                forces(i,2) = forces(i,2) + fij(2)
                forces(i,3) = forces(i,3) + fij(3)

                forces(j,1) = forces(j,1) - fij(1)
                forces(j,2) = forces(j,2) - fij(2)
                forces(j,3) = forces(j,3) - fij(3)
            END DO
        END DO


        ! Triplet loop

        ALLOCATE(neigh_idx(natoms))
        ALLOCATE(neigh_vec(natoms, 3))
        ALLOCATE(neigh_dist(natoms))

        DO i = 1, natoms
            ti = type_ids(i)
            nneigh = 0


            ! Build centered neighbor list around i

            DO j = 1, natoms
                IF (j == i) CYCLE

                CALL displacement_minimum_image(positions(i, :), positions(j, :), &
                                                lattice, inv_lattice, no_pbc, rij_vec)

                rij = norm3(rij_vec)

                IF (rij < rcut_build .AND. rij > 0.0_dp) THEN
                    nneigh = nneigh + 1
                    neigh_idx(nneigh) = j
                    neigh_vec(nneigh, :) = rij_vec
                    neigh_dist(nneigh) = rij
                END IF
            END DO


            ! Unique neighbor pairs (j, k) around center i

            DO a = 1, nneigh - 1
                j = neigh_idx(a)
                tj = type_ids(j)

                rij_vec = neigh_vec(a, :)
                rij = neigh_dist(a)

                DO b = a + 1, nneigh
                    k = neigh_idx(b)
                    tk = type_ids(k)

                    IF (triplet_exists(ti, tj, tk) == 0) CYCLE

                    rik_vec = neigh_vec(b, :)
                    rik = neigh_dist(b)

                    rcut_ij = triplet_rcut_ij(ti, tj, tk)
                    rcut_ik = triplet_rcut_ik(ti, tj, tk)

                    IF (rij >= rcut_ij) CYCLE
                    IF (rik >= rcut_ik) CYCLE

                    CALL sw_triplet_energy_gradients(rij_vec, rik_vec, triplet_params(ti, tj, tk, :), &
                                              rcut_ij, rcut_ik, v3, dh_drij, dh_drik)

                    energy = energy + v3
                    triplet_energy = triplet_energy + v3
                    n_triplets = n_triplets + 1

                    forces(j,1) = forces(j,1) - dh_drij(1)
                    forces(j,2) = forces(j,2) - dh_drij(2)
                    forces(j,3) = forces(j,3) - dh_drij(3)

                    forces(k,1) = forces(k,1) - dh_drik(1)
                    forces(k,2) = forces(k,2) - dh_drik(2)
                    forces(k,3) = forces(k,3) - dh_drik(3)

                    forces(i,1) = forces(i,1) + dh_drij(1) + dh_drik(1)
                    forces(i,2) = forces(i,2) + dh_drij(2) + dh_drik(2)
                    forces(i,3) = forces(i,3) + dh_drij(3) + dh_drik(3)
                END DO
            END DO
        END DO

        DEALLOCATE(neigh_idx)
        DEALLOCATE(neigh_vec)
        DEALLOCATE(neigh_dist)

    END SUBROUTINE sw_eval_kernel


    ! Pair energy and radial derivative dV2/dr
    SUBROUTINE sw_pair_energy_derivative(r, pvec, rcut, v2, dvdr)
        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(IN) :: r
        DOUBLE PRECISION, INTENT(IN) :: pvec(8)
        DOUBLE PRECISION, INTENT(IN) :: rcut
        DOUBLE PRECISION, INTENT(OUT) :: v2
        DOUBLE PRECISION, INTENT(OUT) :: dvdr

        DOUBLE PRECISION :: epsilon, sigma, A, B, p, q, sigma_p, sigma_q
        DOUBLE PRECISION :: sr_p, sr_q, xi, eta, dxi_dr, deta_dr
        DOUBLE PRECISION :: rp1, rq1

        epsilon = pvec(P_EPSILON)
        sigma   = pvec(P_SIGMA)
        A       = pvec(P_A)
        B       = pvec(P_B)
        p       = pvec(P_P)
        q       = pvec(P_Q)
        sigma_p = pvec(P_SIGMA_P)
        sigma_q = pvec(P_SIGMA_Q)

        IF (r >= rcut .OR. r <= 0.0_dp) THEN
            v2 = 0.0_dp
            dvdr = 0.0_dp
            RETURN
        END IF

        sr_p = sigma_p / (r ** p)
        sr_q = sigma_q / (r ** q)
        xi   = B * sr_p - sr_q
        eta  = EXP(sigma / (r - rcut))

        v2 = epsilon * A * xi * eta

        rp1 = r ** (p + 1.0_dp)
        rq1 = r ** (q + 1.0_dp)

        dxi_dr  = -B * p * sigma_p / rp1 + q * sigma_q / rq1
        deta_dr = eta * (-sigma / ((r - rcut) ** 2))

        dvdr = epsilon * A * (dxi_dr * eta + xi * deta_dr)
    END SUBROUTINE sw_pair_energy_derivative


    ! Triplet energy and gradients with respect to rij_vec and rik_vec
    ! Matches Python reference implementation
    SUBROUTINE sw_triplet_energy_gradients(rij_vec, rik_vec, tvec, rcut_ij, rcut_ik, &
                                    v3, dv3_drij, dv3_drik)
        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(IN) :: rij_vec(3), rik_vec(3)
        DOUBLE PRECISION, INTENT(IN) :: tvec(7)
        DOUBLE PRECISION, INTENT(IN) :: rcut_ij, rcut_ik
        DOUBLE PRECISION, INTENT(OUT) :: v3
        DOUBLE PRECISION, INTENT(OUT) :: dv3_drij(3), dv3_drik(3)

        DOUBLE PRECISION :: epsilon, lambda_, gamma_ij, gamma_ik
        DOUBLE PRECISION :: sigma_ij, sigma_ik, cos_theta0
        DOUBLE PRECISION :: rij, rik, cos_theta, delta, expo, pref
        DOUBLE PRECISION :: dcos_drij(3), dcos_drik(3)
        DOUBLE PRECISION :: dexpo_drij(3), dexpo_drik(3)
        DOUBLE PRECISION :: fac_ij, fac_ik

        epsilon    = tvec(T_EPSILON)
        lambda_    = tvec(T_LAMBDA)
        gamma_ij   = tvec(T_GAMMA_IJ)
        gamma_ik   = tvec(T_GAMMA_IK)
        sigma_ij   = tvec(T_SIGMA_IJ)
        sigma_ik   = tvec(T_SIGMA_IK)
        cos_theta0 = tvec(T_COSTHETA0)

        rij = norm3(rij_vec)
        rik = norm3(rik_vec)

        IF (rij >= rcut_ij .OR. rik >= rcut_ik) THEN
            v3 = 0.0_dp
            dv3_drij = 0.0_dp
            dv3_drik = 0.0_dp
            RETURN
        END IF

        IF (rij <= 0.0_dp .OR. rik <= 0.0_dp) THEN
            v3 = 0.0_dp
            dv3_drij = 0.0_dp
            dv3_drik = 0.0_dp
            RETURN
        END IF

        cos_theta = DOT_PRODUCT(rij_vec, rik_vec) / (rij * rik)
        IF (cos_theta > 1.0_dp) cos_theta = 1.0_dp
        IF (cos_theta < -1.0_dp) cos_theta = -1.0_dp

        delta = cos_theta - cos_theta0

        expo = EXP(gamma_ij * sigma_ij / (rij - rcut_ij) + &
                   gamma_ik * sigma_ik / (rik - rcut_ik))

        pref = epsilon * lambda_
        v3 = pref * expo * (delta * delta)

        dcos_drij(1) = rik_vec(1) / (rij * rik) - cos_theta * rij_vec(1) / (rij * rij)
        dcos_drij(2) = rik_vec(2) / (rij * rik) - cos_theta * rij_vec(2) / (rij * rij)
        dcos_drij(3) = rik_vec(3) / (rij * rik) - cos_theta * rij_vec(3) / (rij * rij)

        dcos_drik(1) = rij_vec(1) / (rij * rik) - cos_theta * rik_vec(1) / (rik * rik)
        dcos_drik(2) = rij_vec(2) / (rij * rik) - cos_theta * rik_vec(2) / (rik * rik)
        dcos_drik(3) = rij_vec(3) / (rij * rik) - cos_theta * rik_vec(3) / (rik * rik)

        fac_ij = expo * (-(gamma_ij * sigma_ij / ((rij - rcut_ij) ** 2))) / rij
        fac_ik = expo * (-(gamma_ik * sigma_ik / ((rik - rcut_ik) ** 2))) / rik

        dexpo_drij(1) = fac_ij * rij_vec(1)
        dexpo_drij(2) = fac_ij * rij_vec(2)
        dexpo_drij(3) = fac_ij * rij_vec(3)

        dexpo_drik(1) = fac_ik * rik_vec(1)
        dexpo_drik(2) = fac_ik * rik_vec(2)
        dexpo_drik(3) = fac_ik * rik_vec(3)

        dv3_drij(1) = pref * (dexpo_drij(1) * (delta * delta) + expo * 2.0_dp * delta * dcos_drij(1))
        dv3_drij(2) = pref * (dexpo_drij(2) * (delta * delta) + expo * 2.0_dp * delta * dcos_drij(2))
        dv3_drij(3) = pref * (dexpo_drij(3) * (delta * delta) + expo * 2.0_dp * delta * dcos_drij(3))

        dv3_drik(1) = pref * (dexpo_drik(1) * (delta * delta) + expo * 2.0_dp * delta * dcos_drik(1))
        dv3_drik(2) = pref * (dexpo_drik(2) * (delta * delta) + expo * 2.0_dp * delta * dcos_drik(2))
        dv3_drik(3) = pref * (dexpo_drik(3) * (delta * delta) + expo * 2.0_dp * delta * dcos_drik(3))
    END SUBROUTINE sw_triplet_energy_gradients


    ! Minimum-image displacement
    ! Python convention:
    !   frac = cart @ inv_lattice
    !   cart = frac @ lattice
    SUBROUTINE displacement_minimum_image(ri, rj, lattice, inv_lattice, no_pbc, dr)
        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(IN) :: ri(3), rj(3)
        DOUBLE PRECISION, INTENT(IN) :: lattice(3, 3), inv_lattice(3, 3)
        INTEGER, INTENT(IN) :: no_pbc
        DOUBLE PRECISION, INTENT(OUT) :: dr(3)

        DOUBLE PRECISION :: df(3)

        dr = rj - ri

        IF (no_pbc /= 0) RETURN

        df = MATMUL(TRANSPOSE(inv_lattice), dr)
        df = df - DNINT(df)
        dr = MATMUL(TRANSPOSE(lattice), df)
    END SUBROUTINE displacement_minimum_image


    ! Vector norm
    FUNCTION norm3(v) RESULT(n)
        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(IN) :: v(3)
        DOUBLE PRECISION :: n

        n = SQRT(DOT_PRODUCT(v, v))
    END FUNCTION norm3


    ! Cell volume
    FUNCTION cell_volume(lattice) RESULT(vol)
        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(IN) :: lattice(3, 3)
        DOUBLE PRECISION :: vol
        DOUBLE PRECISION :: c(3)

        c(1) = lattice(2,2) * lattice(3,3) - lattice(2,3) * lattice(3,2)
        c(2) = lattice(2,3) * lattice(3,1) - lattice(2,1) * lattice(3,3)
        c(3) = lattice(2,1) * lattice(3,2) - lattice(2,2) * lattice(3,1)

        vol = ABS(lattice(1,1) * c(1) + lattice(1,2) * c(2) + lattice(1,3) * c(3))
    END FUNCTION cell_volume

END MODULE fortran_backend
