import Computations_iV as calc
import FilePaths as fp
import Tabulates as tb
import Plots as plt
import Anicalcs as ani
import numpy as np
import time
import Test_MFPT as test
import random
import math


if __name__ == "__main__":

    # Rings: radial curves (m), Rays: angular rays (n)
    rings = 16
    rays = 16

    # Diffusion coefficient
    d = 1
    # Domain Radius
    r = 1
    # Run time
    t = 1

    # Constraints placed upon calculations of currents; diffusive and advective layers respectively
    a = 0
    b = 0
    v = -10

    # For general use cases in plotting and tabulation operations
    target_angle = math.ceil(rays / 2)
    target_ring = math.ceil(rings / 2)

    # New content as 8/12/24, 2:44PM
    # Number of iterations for the calculation of Mean Flow Passage Time
    w_iterations = 10
    # The steps in between each gradually increasing value of W
    w_step = 1
    # The extension factor at which the Mean Flow Passage Time will be computed under
    glb_time_ex = 20

    # Container for the underlying mean flow passage time values; graphing operations
    # Container for the underlying values of w; graphing operations

    # 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3, 10^4
    w_choices = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    w_container = np.zeros([len(w_choices)])
    m_f_p_t_container = np.zeros([len(w_choices)])

    tube_config_i = [0, 4, 8, 12]
    tube_config_ii = [0, 2, 4, 6, 8, 10, 12, 14]
    tube_config_iii = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    tube_placement_container = [tube_config_i, tube_config_ii, tube_config_iii]

    # Compute the plot for the mean flow passage time
    calculate_plot_m_f_p_t = True

    if calculate_plot_m_f_p_t:

        for j in range(len(tube_placement_container)):
            tube_placement_container[j] = np.unique(np.sort(tube_placement_container[j]))
            print(tube_placement_container[j])
            for i in range(len(w_choices)):
                w = w_choices[i] * w_step
                print(f'w : {w}')
                # glb_mx_ml_time : Global max mass loss rate exact time

                # Computing the exact time for which the global max in the mass loss rate graph is attained
                glb_mx_ml_time = calc.solve(rings, rays, r, d, t, False, 1, w, w, v, tube_placement_container[j], True, False, False, False)
                print(f'Time at which the global max was achieved: {glb_mx_ml_time}')

                # New time computed using the specified time-extension factor
                glb_mx_ml_time *= glb_time_ex

                # Computing the mean flow passage time associated to the new time and the current value for W
                m_f_p_t = calc.solve(rings, rays, r, d, glb_mx_ml_time, True, 1, w, w, v, tube_placement_container[j], False, False, True, False)

                print(f'MFPT: {m_f_p_t}')
                w_container[i] = w
                m_f_p_t_container[i] = m_f_p_t

            tb.tab_mfpt_by_w(m_f_p_t_container, w_container, len(tube_placement_container[j]), fp.mfpt_fp)

    # for i in range(len(w_choices)):
    #     w = w_choices[i] * w_step
    #     print(f'w : {w}')
    #     # glb_mx_ml_time : Global max mass loss rate exact time
    #
    #     # Computing the exact time for which the global max in the mass loss rate graph is attained
    #     glb_mx_ml_time = calc.solve(rings, rays, r, d, t, False, 1, w, w, v, tube_placement_container[0], True, False,
    #                                 False, False)
    #     print(f'Time at which the global max was achieved: {glb_mx_ml_time}')
    #
    #     # New time computed using the specified time-extension factor
    #     glb_mx_ml_time *= glb_time_ex
    #
    #     # Computing the mean flow passage time associated to the new time and the current value for W
    #     m_f_p_t = calc.solve(rings, rays, r, d, glb_mx_ml_time, True, 1, w, w, v, tube_placement_container[0], False,
    #                          False, True, False)
    #
    #     print(f'MFPT: {m_f_p_t}')
    #     w_container[i] = w
    #     m_f_p_t_container[i] = m_f_p_t
    #
    # tb.tab_mfpt_by_w(m_f_p_t_container, w_container, len(tube_placement_container[ t ]), fp.mfpt_fp)
    #




