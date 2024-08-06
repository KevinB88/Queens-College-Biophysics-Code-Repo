# import calc
import calc_version_ii as calc
import tabulate_functions as tb
import plot_functions as plt
from remote import upload
from shut_off_op import pc_sleep
import math
import time
import Anicalcs

if __name__ == "__main__":

    rings = 15
    rays = 15
    # Diffusion coefficient
    d = 1
    # Domain Radius
    r = 1
    # Run time
    t = 1

    a = 500
    b = 500
    v = -1

    # for tabulate and plot operations
    tube_placement = math.ceil(rays / 2)
    angle = math.ceil(rays / 2)
    ring = math.ceil(rings / 2)

    # base_mass_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\mass_data\\base_mass"
    mass_loss_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\mass_data\\mass_loss"
    phi_angular_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\angular"
    # phi_central_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\central"
    phi_radial_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\radial"
    # rho_filepath_table = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\rho_densities\\table"
    # rho_filepath_plot = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\rho_densitiess\\plot"
    # animation_filepath = "N:\\Queens College 2024\\Research\Theoretical-Biophysics\\Main\\tabulate\\animation"
    remote = "N:\\Queens College 2024\\Research\\Theoretical-Biophysics\\Main\\tabulate\\transfer-remote"
    # generic = "N:\\Queens College 2024\\Research\\Theoretical-Biophysics\Main\\tabulate\Generic"

    compare = "N:\\Queens College 2024\\Research\\Theoretical-Biophysics\\Main\\tabulate\\mass_data\\mass_loss\\comparison"

    # plot1 = "N:\\Queens College 2024\\Research\\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\radial\\density_across_radius_data_date20240806_074734_time1_angle8_w=0.csv"
    # plot2 = "N:\\Queens College 2024\\Research\\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\radial\\density_across_radius_data_date20240806_075043_time1_angle8_w=1.csv"
    # plot3 = "N:\\Queens College 2024\\Research\\Theoretical-Biophysics\\Main\\tabulate\\phi_densities\\radial\\density_across_radius_data_date20240806_075412_time1_angle8_w=100.csv"
    #
    # container = [plot1, plot2, plot3]
    #
    # plt.plot_multiple_general(container, ['w=0', 'w=1', 'w=100'], True, phi_radial_filepath, True, 'R', 'D', f'Phi versus Radius at angle(n)=8, and time={t}')

    #
    density_table, center_table, mass_table, mass_loss_i, mass_loss_ii, delta_time, delta_radius, rho_table, info_dict = calc.solve_ii(rings, rays, r, d, t, True, 1, a, b, v, tube_placement)
    # #
    tb.tabulate_mass_loss(mass_loss_i, mass_loss_filepath, t, delta_time, a)
    time.sleep(1)
    tb.tabulate_mass_loss(mass_loss_ii, mass_loss_filepath, t, delta_time, a)
    # time.sleep(1)
    #
    # tb.tabulate_densities(density_table, phi_radial_filepath, 0, t, delta_radius, a)
    # time.sleep(1)
    # tb.tabulate_densities(density_table, phi_radial_filepath, tube_placement, t, delta_radius, a)

    # ensure a comparison between the graphs of mass loss J and mass loss derivative






