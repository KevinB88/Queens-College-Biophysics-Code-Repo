import computations as comp
import numpy as np
import math
import random as r_gen
from datetime import datetime

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


if __name__ == "__main__":

    # radius of the underlying domain relative to the number of rings and rays (rays)
    radius = 1

    rings = 12
    rays = 12
    # simulation time (sim_time)
    sim_time = 0.14
    # diffusion coefficient (d_c)
    d_c = 1

    # diffusive velocity across the domain,
    v = -10

    tube_placements = [0, 3, 6, 9]
    tube_placements = np.sort(np.unique(tube_placements))

    # implement a feature to randomize the tube-placements, this could potentially be added within the U.I update

    # The following denote the lower magnitude bound (l_mag_bound), and upper magnitude bound (u_mag_bound)
    # These bounds are used for the for-loop below to populate a list an array with different orders of magnitude equivalent to 'w'
    # 'w' : w = a = b, a general switching rate between DL and the AL, more specifically between microtubules and the cytoplasm
    l_mag_bound = -3
    u_mag_bound = 3

    w_list = []

    for i in range(int(l_mag_bound), int(u_mag_bound+1)):
        w_list.append(10 ** i)

    mfpt_list = []

    # quantify(rings, rays, radius, d_c, sim_time, tube_placements)
    tests = 10

    for w in range(len(w_list)):
        curr_w = w_list[w]
        t_prime = comp.solve(rings, rays, radius, d_c, sim_time, False, 1, curr_w, curr_w, v, tube_placements, ['global-max'])
        print(f'Global max acquired at time: {t_prime} for w={w_list[w]} : {w}')


    # for t in range(tests):
    #
    #     random_size = r_gen.randint(1, rays)
    #     tube_placements = np.unique(np.sort(np.random.randint(0, rays, size=random_size)))
    #
    #     quantify(rings, rays, radius, d_c, sim_time, tube_placements)
    #
    #     for w in range(len(w_list)):
    #
    #         curr_w = w_list[w]
    #
    #         t_prime = comp.solve(rings, rays, radius, d_c, sim_time, False, 1, curr_w, curr_w, v, tube_placements, ['global-max'])
    #
    #         print()
    #
    #         print(f'Global max acquired at time: {t_prime} for w={w_list[w]} : {w}')
