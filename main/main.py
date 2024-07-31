# import calc
import calc_version_ii as calc
import tabulate_functions as tb
import plot_functions as plt
import numpy as np
from playsound import playsound

if __name__ == "__main__":

    rings = 24
    rays = 24
    # Diffusion coefficient
    d = 1
    # Domain Radius
    r = 1
    # Run time
    t = 0.1

    a = 0.01
    b = 0.01
    v = -1

    tube_placement = 12

    domain_density_filepath = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_density"
    domain_density_angle_plot_filepath = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_density/density_angle_plots"
    domain_microtubule_filepath = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_microtubule_density"
    mass_table_filepath = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_mass"
    domain_microtubule_plot_filepath = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_microtubule_density/plot"
    domain_angles_density = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_density/phi_versus_angles"


    values_w = [0, 0.01, 0.1, 1]

    # plt.plot_multiple_general([w_0, w_0_0_1, w_0_1, w_1], values_w)

    # UNCOMMENT THE FOLLOWING BELOW !!IMPORTANT
    # 7/30/24
    density_table, center_table, mass_table, mass_loss_i, mass_loss_ii, delta_time, delta_radius, rho_table, info_dict = calc.solve_ii(
        rings, rays, r, d, t, True, 1, a, b, v, tube_placement)

    playsound(
        "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VI/Sound/609336__kenneth_cooney__completed.wav")

    # time_steps = len(density_table)
    #
    # tb.tabulate_density_angles(density_table, domain_density_angle_plot_filepath, t, tube_placement, 12)

    # def tabulate_rho_density(container, filepath, time, delta_radius):
    # tb.tabulate_rho_density(rho_table, domain_microtubule_filepath, t, delta_radius)
    # plt.plot_rho_density(rho_table, r, t, info_dict, domain_microtubule_plot_filepath, True, True)
    #
    tb.tabulate_densities(density_table, domain_density_filepath, 0, t, delta_radius)

    # plt.plot_phi_angle(density_table, 12, t, False)
    # plt.plot_final_density(density_table, r, 12)

    # plt.plot_phi_angle(density_table, 1, t, False)
    # plt.plot_phi_angle(density_table, 10, t, False)



   # w_0 = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_density/density_angle_plots/angle_density_20240730_153410_time0.1_w=0.csv"
   #  w_0_0_1 = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_density/density_angle_plots/angle_density_20240730_153600_time0.1_w=0.01.csv"
   #  w_0_1 = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_density/density_angle_plots/angle_density_20240730_153732_time0.1.csv"
   #  w_1 = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/domain_density/density_angle_plots/angle_density_20240730_154402_time0.1.csv"