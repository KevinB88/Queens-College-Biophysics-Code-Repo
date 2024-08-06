import Computations_i as calculate_i
import Computations_ii as calculate_ii
import FilePaths as fp
import Tabulates as tb
import Plots as plt
import numpy as np
import time
import math


if __name__ == "__main__":

    # Rings: radial curves (m), Rays: angular rays (n)
    rings = 15
    rays = 15

    # Diffusion coefficient
    d = 1
    # Domain Radius
    r = 1
    # Run time
    t = 0.1

    # Constraints placed upon calculations of currents; diffusive and advective layers respectively
    a = 1
    b = 1
    v = -1

    # For general use cases in plotting and tabulation operations
    tube_placement = math.ceil(rays / 2)
    angle = math.ceil(rays / 2)
    ring = math.ceil(rings / 2)

    # !NEW FEATURE! This allows the user to place multiple microtubules by choosing candidate angles
    tube_placements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # Sorts the provided list of angles, and removes any duplicate angles
    tube_placements = np.unique(np.sort(tube_placements))

    # !This version only calculates the microtubule across a single angle!
    # density_table, center_table, mass_table, mass_loss_i, mass_loss_ii, delta_time, delta_radius, rho_table, info_dict = calculate_i.solve(rings, rays, r, d, t, True, 1, a, b, v, tube_placement)

    # !This version generalizes microtubule calculation across the domain!
    phi_dense, phi_center, mass_table, mass_loss_j_r_r, mass_loss_delta, delta_time, delta_radius, rho_dict, info_dict = calculate_ii.solve(rings, rays, r, d, t, True, 1, a, b, v, tube_placements)

    plt.plt_mass_base(mass_table, t, False, fp.base_mass_plt_fp, True)
