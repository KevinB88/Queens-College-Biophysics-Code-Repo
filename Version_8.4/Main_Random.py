import Computations_i as calculate_i
import Computations_ii as calculate_ii
import Computations_iii as calculate_iii
import FilePaths as fp
import Tabulates as tb
import Plots as plt
import numpy as np
import time
import random
import math


if __name__ == "__main__":

    # Rings: radial curves (m), Rays: angular rays (n)
    rings = 20
    rays = 20

    # Diffusion coefficient
    d = 1
    # Domain Radius
    r = 1
    # Run time
    t = 0.04

    # Constraints placed upon calculations of currents; diffusive and advective layers respectively
    a = 500
    b = 500
    v = -1

    # w_list = [0, 1, 10, 15, 20, 30, 50, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
    # m_f_p_t_list = np.zeros([len(w_list)])

    # For general use cases in plotting and tabulation operations
    tube_placement = math.ceil(rays / 2)
    angle = math.ceil(rays / 2)
    ring = math.ceil(rings / 2)

    while True:

        # !NEW FEATURE! This allows the user to place multiple microtubules by choosing candidate angles
        # tube_placements = [0, 3, 6, 9, 12]
        tube_placements = np.random.randint(0, rays-1, random.randint(1, rays-1))
        # Sorts the provided list of angles, and removes any duplicate angles
        tube_placements = np.unique(np.sort(tube_placements))

        # !This version only calculates the microtubule across a single angle!
        # density_table, center_table, mass_table, mass_loss_i, mass_loss_ii, delta_time, delta_radius, rho_table, info_dict = calculate_i.solve(rings, rays, r, d, t, True, 1, a, b, v, tube_placement)
        #
        # !This version generalizes microtubule calculation across the domain!
        # phi_dense, phi_center, mass_table, mass_loss_j_r_r, mass_loss_delta, delta_time, delta_radius, rho_dict, info_dict, rho_mass, phi_mass = calculate_ii.solve(rings, rays, r, d, t, True, 1, a, b, v, tube_placements)

        phi_dense, phi_center, mass_table, mass_loss_j_r_r, mass_loss_delta, delta_time, delta_radius, rho_dict, info_dict, rho_mass, phi_mass, mean_first_passage_time = calculate_iii.solve(
            rings, rays, r, d, t, True, 1, a, b, v, tube_placements)

        # for i in range(len(w_list)):
        #     phi_dense, phi_center, mass_table, mass_loss_j_r_r, mass_loss_delta, delta_time, delta_radius, rho_dict, info_dict, rho_mass, phi_mass, mean_first_passage_time = calculate_iii.solve(
        #         rings, rays, r, d, t, True, 1, w_list[i], w_list[i], v, tube_placements)
        #     m_f_p_t_list[i] = mean_first_passage_time
        #
        # for j in range(len(w_list)):
        #     print(f'{w_list[j]} : {m_f_p_t_list[j]}')
        #
        # # m_f_p_t_list = [11.776070268917758, 32.522618848726225, 55.323106762008834, 54.87931317259481, 54.24290552270968, 52.20753657801399,51.51177704285594 ,51.3750926122132 ,51.26403112295243,50.92189015975909,50.74589946563538 ,50.42113251638133]
        #
        # # tb.tab_mass_base(phi_mass, fp.base_mass_fp, t, delta_time, a)
        # plt.plt_mean_passage(w_list, m_f_p_t_list, False)
        # plt.plt_mean_passage(w_list, m_f_p_t_list, True)

        plt.plt_mass_base(mass_table, t, False, fp.base_mass_plt_fp, True, f'Microtubule placements (angles): {tube_placements}')


