# import calc
import calc_version_ii as calc
import tabulate_functions as tb
import plot_functions as plt
from remote import upload
from shut_off_op import pc_sleep
import math
import time

if __name__ == "__main__":

    rings = 15
    rays = 15
    # Diffusion coefficient
    d = 1
    # Domain Radius
    r = 1
    # Run time
    t = 0.1

    a = 100
    b = 100
    v = -1

    # for tabulate and plot operations
    tube_placement = math.ceil(rays / 2)
    angle = math.ceil(rays / 2)
    ring = math.ceil(rings / 2)

    # base_mass_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\mass_data\\base_mass"
    # mass_loss_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\mass_data\\mass_loss"
    # phi_angular_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\angular"
    # phi_central_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\central"
    # phi_radial_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\radial"
    # rho_filepath_table = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\rho_densities\\table"
    rho_filepath_plot = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\rho_densitiess\\plot"
    # animation_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\animation"
    # remote = "N:\\Queens College 2024\\Research\\Theoretical-Biophysics\\Main\\tabulate\\transfer-remote"
    # generic = "N:\\Queens College 2024\\Research\\Theoretical-Biophysics\Main\\tabulate\Generic"

    density_table, center_table, mass_table, mass_loss_i, mass_loss_ii, delta_time, delta_radius, rho_table, info_dict = calc.solve_ii(rings, rays, r, d, t, True, 1, a, b, v, tube_placement)

    plt.plot_rho_versus_radius(rho_table, r)

    # tb.tabulate_mass(mass_table, base_mass_filepath, t, delta_time)
    # tb.tabulate_mass_loss(mass_loss_ii, remote, t, delta_time)
    #
    # tb.tabulate_densities(density_table, phi_radial_filepath, 0, t, delta_radius)
    # tb.tabulate_densities(density_table, phi_radial_filepath, angle, t, delta_radius)
    # tb.tabulate_central(center_table, remote, t, delta_time)
    # tb.tabulate_density_angles(density_table, remote, t, ring)

    tb.tabulate_rho_density(rho_table, rho_filepath_plot, t, delta_radius)

    # plt.plot_mass(mass_table, t, True, base_mass_filepath, False)
    #
    # plt.plot_mass_loss(mass_loss_ii, t, True, remote, False)
    #
    # plt.plot_final_density(density_table, r, 0, True, remote, t, False)
    #
    # plt.plot_final_density(density_table, r, angle, True, remote, t, False)
    #
    # plt.plot_phi_angle(density_table, ring, t, False, True, phi_angular_filepath, True)
    #
    # plt.plot_central_density(center_table, t, True, remote, False)
    #
    # plt.plot_rho_density(rho_table, r, t, info_dict, rho_filepath_plot, False, False, True)
    #
    # time_steps = len(density_table)

    # plt.plot_density_heat_map_discrete_animated(density_table, center_table, time_steps, animation_filepath)

    # upload("N:\\Queens College 2024\\Research\\Theoretical-Biophysics\\Main\\tabulate\\transfer-remote")
    #
    # time.sleep(60)
    #
    # pc_sleep()
