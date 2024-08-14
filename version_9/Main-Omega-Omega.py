import multiprocessing as mp
import Computations_V as calc
import Tabulates as tb
import Plots as plt
import math
import time as clock
import FilePaths as fp
import numpy as np


def run_solve(output_specification):

    start = clock.time()
    rings = 32
    rays = 32
    radius = 1
    diffusion_coefficient = 1
    time_k = 1
    w = 100
    v = -10

    tube_placements = [0, 8, 16, 24]
    # tube_placements = [0, 2, 4, 6, 8]

    # specification must be made on which microtubules should be inspected

    if output_specification.lower() == 'advective_layer':
        output_container, info_dict = calc.solve(rings, rays, radius, diffusion_coefficient, time_k, True, 1, w, w, v, tube_placements, output_specification, True)

        for tube_index in range(len(tube_placements)):
            steps = len(output_container)
            rings = len(output_container[len(output_container) - 1])
            rep_microtubule_state = np.zeros([steps, rings])
            final_microtubule_state = output_container[steps-1]

            current_angle = tube_placements[tube_index]
            for m in range(rings):
                rep_microtubule_state[steps-1][m] = final_microtubule_state[m][current_angle]

            d_radius = radius / rings
            tb.tab_rho_rad(rep_microtubule_state, fp.rho_rad_fp, time_k, d_radius, w, current_angle)
            plt.plt_rho_rad(rep_microtubule_state, radius, time_k, info_dict, fp.rho_raf_plt, True, False, False, current_angle)

        end = clock.time()
        duration = end - start
        print(f'Duration: {duration:.4f} seconds: {output_specification}')
    else:
        output_container = calc.solve(rings, rays, radius, diffusion_coefficient, time_k, True, 1, w, w, v, tube_placements, output_specification, False)

    if output_specification.lower() == 'diffusive_layer_time_0.2':
        d_radius = radius / rings
        d_theta = ((2 * math.pi) / rays)
        delta_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * diffusion_coefficient)
        delta_radius = radius / rings

        tb.tab_phi_rad(output_container, fp.phi_rad_fp, math.ceil(rays/2), time_k, delta_radius, w)
        plt.plt_phi_rad(output_container, radius, math.ceil(rays/2), True, fp.phi_rad_plt_fp, time_k, False, delta_time, 0.2)

        tb.tab_phi_ang(output_container, fp.phi_ang_fp, time_k, math.ceil(rings/2))
        plt.plt_phi_ang(output_container, math.ceil(rings/2), time_k, False, True, fp.phi_ang_plt_fp, False)
        end = clock.time()
        duration = end - start
        print(f'Duration: {duration:.4f} seconds')

    if output_specification.lower() == 'diffusive_layer_time_0.5':
        d_radius = radius / rings
        d_theta = ((2 * math.pi) / rays)
        delta_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (
                    2 * diffusion_coefficient)
        delta_radius = radius / rings

        tb.tab_phi_rad(output_container, fp.phi_rad_fp, math.ceil(rays / 2), time_k, delta_radius, w)
        plt.plt_phi_rad(output_container, radius, math.ceil(rays / 2), True, fp.phi_rad_plt_fp, time_k, False,
                        delta_time, 0.5)

        tb.tab_phi_ang(output_container, fp.phi_ang_fp, time_k, math.ceil(rings / 2))
        plt.plt_phi_ang(output_container, math.ceil(rings / 2), time_k, False, True, fp.phi_ang_plt_fp, False)
        end = clock.time()
        duration = end - start
        print(f'Duration: {duration:.4f} seconds')

    if output_specification.lower() == 'diffusive_layer_time_1':
        d_radius = radius / rings
        d_theta = ((2 * math.pi) / rays)
        delta_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (
                    2 * diffusion_coefficient)
        delta_radius = radius / rings

        tb.tab_phi_rad(output_container, fp.phi_rad_fp, math.ceil(rays / 2), time_k, delta_radius, w)
        plt.plt_phi_rad(output_container, radius, math.ceil(rays / 2), True, fp.phi_rad_plt_fp, time_k, False,
                        delta_time, 1)

        tb.tab_phi_ang(output_container, fp.phi_ang_fp, time_k, math.ceil(rings / 2))
        plt.plt_phi_ang(output_container, math.ceil(rings / 2), time_k, False, True, fp.phi_ang_plt_fp, False)
        end = clock.time()
        duration = end - start
        print(f'Duration: {duration:.4f} seconds')

    if output_specification.lower() == 'central_patch':
        d_radius = radius / rings
        d_theta = ((2 * math.pi) / rays)
        delta_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * diffusion_coefficient)

        tb.tab_phi_cent(output_container, fp.phi_cent_fp, time_k, delta_time)
        plt.plt_phi_center(output_container, time_k, True, fp.phi_cent_plt_fp, False)
        end = clock.time()
        duration = end - start
        print(f'Duration: {duration:.4f} seconds')

    if output_specification.lower() == 'total_mass':
        d_radius = radius / rings
        d_theta = ((2 * math.pi) / rays)
        delta_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * diffusion_coefficient)

        tb.tab_mass_base(output_container, fp.base_mass_fp, time_k, delta_time, w)
        plt.plt_mass_base(output_container, time_k, True, fp.base_mass_plt_fp, False, 'Total mass versus time')
        end = clock.time()
        duration = end - start
        print(f'Duration: {duration:.4f} seconds')

    if output_specification.lower() == 'mass_loss_rate_j_r_r' or output_specification.lower() == 'mass_loss_rate_derivative':
        d_radius = radius / rings
        d_theta = ((2 * math.pi) / rays)
        delta_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * diffusion_coefficient)

        if output_specification == 'mass_loss_rate_j_r_r':
            calculation_type = 'mass_loss_rate_j_r_r'
        else:
            calculation_type = 'mass_loss_rate_derivative'

        tb.tab_mass_loss(output_container, fp.mass_loss_fp, time_k, delta_time, w, calculation_type)
        plt.plt_mass_loss(output_container, time_k, True, fp.mass_loss_plt_fp, False)
        end = clock.time()
        duration = end - start
        print(f'Duration: {duration:.4f} seconds')


if __name__ == "__main__":
    output_specifications = ['diffusive_layer_time_0.2', 'diffusive_layer_time_0.5', 'diffusive_layer_time_1']

    with mp.Pool(processes=8) as pool:
        pool.map(run_solve, output_specifications)

# 'total_mass', 'mass_loss_rate_j_r_r', 'mass_loss_rate_derivative'

