import computations_op1 as comp
import numpy as np

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

if __name__ == "__main__":

    # radius of the underlying domain relative to the number of rings and rays (rays)
    radius = 1

    rings = 12
    rays = 12
    # simulation time (sim_time)
    sim_time = 1
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
    l_mag_bound = -2
    u_mag_bound = 4

    w_list = []

    for i in range(int(l_mag_bound), int(u_mag_bound + 1)):
        w_list.append(10 ** i)

    mfpt_list = []
    # time extension factor (t_ext_fact)
    t_ext_fact = 5

    output_specification = ['global-max', 'mean-flow-passage-time']

    for w in range(len(w_list)):
        curr_w = w_list[w]
        mfpt_curr, info_dict = comp.solve(rings, rays, radius, d_c, sim_time, False, 1, curr_w, curr_w, v, tube_placements, t_ext_fact, output_specification)

        mfpt_list.append(mfpt_curr)

    print(f'MFPT list: {mfpt_list}')
    print(f'W list: {w_list}')


