import computations as comp
import numpy as np
import math
import random as r_gen
from datetime import datetime
import plots as plt
import FilePaths as fp
import tabulates as tb

'''
    Computation for MFPT:
    
    
    For some number of microtubules, and for some value of w=a=b, that is the switching rates between the diffusive layer (DL), 
    and the advective layer (AL). This will be done for some value v denoting velocity. 
    
    Run the algorithm and compute for the global-max of the mass function/table and return the corresponding time.
    
    Using this returned time denoted by t-prime, multiply this time by an temporal-extension-factor, decided by the user, and 
    then re-run the algorithm for the computed time and return the MFPT, store this value to an array.
    
    Repeat the process for the next value of 'w' within the list of 'w' values.
    
    
    *potential optimizations/safety features: ensure that when the MFPT is returned, it immediately updates the CSV 
    (preventing the loss of data, and avoiding the need to rely for the algorithm to fully complete after data is collected) 
'''


def quantify(domain_rings, domain_rays, r, d, t, domain_tube_placements):
    d_radius = r / domain_rings
    d_theta = ((2 * math.pi) / domain_rays)
    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * d)
    time_steps = math.ceil(t / d_time)
    phi_center = 1 / (math.pi * (d_radius * d_radius))

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Current date of operation: {current_time}')
    print(f'Radial Curves: {rings}')
    print(f'Angular Rays: {rays}')
    print(f'Domain Radius: {r}')
    print(f'Diffusion Coefficient: {d}')
    print(f'Run Time: {t}')
    print(f'Delta Radius: {d_radius}')
    print(f'Delta Theta: {d_theta}')
    print(f'Delta Time: {d_time}')
    print(f'Time steps: {time_steps}')
    print(f'Initial Central Density: {phi_center}')
    print(f'Velocity: {v}')
    print(f'Microtubule placements: {domain_tube_placements}')


'''
    9/17/24 : 10:46 PM, Algorithm ran for the following parameters:
    rings: 32, rays: 32, sim_time = 0.15, d_c = 1, v=-10, w=100
'''

if __name__ == "__main__":

    # radius of the underlying domain relative to the number of rings and rays (rays)
    radius = 1
    rings = 32
    rays = 32
    # simulation time (sim_time)
    # sim_time = 0.11655360345715476 * 20
    sim_time = 0.22320725697356533 * 20
    # sim_time = 1
    # diffusion coefficient (d_c)
    d_c = 1

    # diffusive velocity across the domain,
    v = -20
    w = 1000
    #
    tube_placements = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    # tube_placements = [0, 1]
    tube_placements = np.sort(np.unique(tube_placements))

    '''
    UPDATE: 9/17/24 : 10:29 PM 
    
    The current implementation does not correctly return all the containers from after a run of the algorithm.
    
    Perhaps try returning a container of all the relevant containers specified in the output specification vector.
    
    Then specify which container to use within for plots and CSVs using indices
    
    '''

    mass_loss, delta_time, delta_radius, diffusive_layer = comp.solve(rings, rays, radius, d_c, sim_time, True, 1, w, w, v, tube_placements, ['mass_loss_rate_j_r_r', 'global-max'])

    # add the feature to determine when the peak is attained within the graph
    #

    # filename = '/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Fall-Software/algo-version-9.1/output/MFPT-related/Current-mass-loss/mass_loss_data_20240918_034548_2.3310720691430955_w1000_J_R_R.csv'
    #
    # plt.plot_generic_csv(filename, 'T', 'M', 'Time', 'Mass loss rate', f'Mass loss rate versus time: {sim_time}')

    tb.tab_mass_loss(mass_loss[0], fp.mass_loss_fp, sim_time, delta_time, w, 'J_R_R')
    tb.tab_phi_rad(diffusive_layer, fp.phi_rad_plt_fp, 7, sim_time, delta_radius, w)
    plt.plt_mass_loss(mass_loss[0], sim_time, True, fp.mass_loss_fp, True)
    plt.plt_phi_rad(diffusive_layer, radius, 7, True, fp.phi_rad_plt_fp, sim_time, True, delta_time, 0.1)
    plt.plt_phi_rad(diffusive_layer, radius, 7, True, fp.phi_rad_plt_fp, sim_time, True, delta_time, 1)


