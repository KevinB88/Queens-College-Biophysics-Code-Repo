import math
import prints as pr
import numpy as np

# this version is slightly optimized with respect to the MFPT calculations: last edited: 9/13/24 @ 4:26 PM

# !This version generalizes microtubule calculation across the domain!


# Update contents at the diffusive layer (Particle density computation across the domain), returns a floating point number
def u_density(phi, k, m, n, d_radius, d_theta, d_time, central, rings, rho, dict_index, a, b, tube_placements):

    current_density = phi[k][m][n]

    component_a = ((m+2) * j_r_r(phi, k, m, n, d_radius, rings)) - ((m+1) * j_l_r(phi, k, m, n, d_radius, central))

    component_a *= d_time / ((m+1) * d_radius)

    component_b = (j_r_t(phi, k, m, n, d_radius, d_theta)) - (j_l_t(phi, k, m, n, d_radius, d_theta))
    component_b *= d_time / ((m+1) * d_radius * d_theta)

    if n == tube_placements[dict_index]:  # if the current angle 'n' equates to the angle for which the microtubule is positioned at
        component_c = (a * phi[k][m][n]) * d_time - (((b * rho[k][m][n]) * d_time) / ((m+1) * d_radius * d_theta))
        # print(f'k={k}, m={m}, n={n}, dict_index={dict_index}')
        #
        # print(f'Dict index{dict_index}, Comp c: {component_c}, removed at angle {n}')
        # if m == 0:
        #     print(f' Component C at segment: {m} and at angle {n} : {component_c}')
    else:
        component_c = 0

    return current_density - component_a - component_b - component_c


# Update contents at advective layer (Particle density computation along a microtubule), returns a floating point number
def u_tube(rho, phi, k, m, n, a, b, v, d_time, d_radius, d_theta):
    j_l = v * rho[k][m][n]
    if m == len(phi[k][m]) - 1:
        j_r = 0
    else:
        j_r = v * rho[k][m+1][n]

    return rho[k][m][n] - ((j_r - j_l) / d_radius) * d_time + (a * phi[k][m][n] * (m+1) * d_radius * d_theta) * d_time - b * rho[k][m][n] * d_time


# Update the central patch, returns a floating point value
def u_center(phi, k, d_radius, d_theta, d_time, curr_central, rho, tube_placements, v):
    total_sum = 0
    for n in range(len(phi[k][0])):
        total_sum += j_l_r(phi, k, 0, n, d_radius, curr_central)

    total_sum *= (d_theta * d_time) / (math.pi * d_radius)
    diffusive_sum = curr_central - total_sum

    advective_sum = 0

    # Necessary for acquiring the associated angle at a microtubule via index from the list of keys from the dictionary

    for i in range(len(tube_placements)):
        angle = tube_placements[i]
        j_l = rho[k][0][angle] * v
        advective_sum += (abs(j_l) * d_time) / (math.pi * d_radius * d_radius)

    return diffusive_sum + advective_sum
    # return diffusive_sum


# calculate for total mass across domain, returns a floating point number
def calc_mass(phi, rho, k, d_radius, d_theta, curr_central, rings, rays, rho_mass, phi_mass, tube_placements):
    mass = 0
    for m in range(rings):
        for n in range(rays):
            mass += phi[k][m][n] * (m+1)
    mass *= (d_radius * d_radius) * d_theta

    phi_mass[k] = mass + (curr_central * math.pi * d_radius * d_radius)

    microtubule_mass = 0

    for i in range(len(tube_placements)):
        angle = tube_placements[i]
        for m in range(rings):
            microtubule_mass += rho[k][m][angle] * d_radius

    rho_mass[k] = microtubule_mass

    # print(f'{phi_mass[k]} + {rho_mass[k]} = {phi_mass[k] + rho_mass[k]}')

    return (curr_central * math.pi * d_radius * d_radius) + mass + microtubule_mass
    # return (curr_central * math.pi * d_radius * d_radius) + mass


# calculate mass loss using the J_R_R scheme, returns a floating point number
def calc_loss_mass_j_r_r(phi, k, d_radius, d_theta, rings, rays):
    total_sum = 0
    for n in range(rays):
        total_sum += j_r_r(phi, k, rings-2, n, d_radius, 0)
    total_sum *= rings * d_radius * d_theta

    return total_sum


# calculate mass loss using the derivative scheme, returns a floating point number
def calc_loss_mass_derivative(mass_container, d_time):
    array = np.zeros([len(mass_container) - 1])
    for k in range(1, len(mass_container)):
        array[k-1] = (mass_container[k-1] - mass_container[k]) / d_time
    return array


# J Right Radius
def j_r_r(phi, k, m, n, d_radius, rings):
    curr_ring = phi[k][m][n]
    if m == rings - 1:
        next_ring = 0
    else:
        next_ring = phi[k][m+1][n]
    return -1 * ((next_ring - curr_ring) / d_radius)


# J Left Radius
def j_l_r(phi, k, m, n, d_radius, central):
    curr_ring = phi[k][m][n]
    if m == 0:
        prev_ring = central
    else:
        prev_ring = phi[k][m-1][n]
    return -1 * ((curr_ring - prev_ring) / d_radius)


# J Right Theta
def j_r_t(phi, k, m, n, d_radius, d_theta):
    b = len(phi[k][m])
    return -1 * (phi[k][m][(n+1) % b] - phi[k][m][n]) / ((m+1) * d_radius * d_theta)


# J Left Theta
def j_l_t(phi, k, m, n, d_radius, d_theta):
    b = len(phi[k][m])
    return -1 * (phi[k][m][n] - phi[k][m][(n-1) % b]) / ((m+1) * d_radius * d_theta)


'''
rings   = Radial curves 
rays    = Angular rays
r       = Domain radius
d       = Diffusion coefficient
t       = Run time
q       = Passing a boolean value to determine if the user would like to print initial quantities
p       = Progression frequency for print operations
a       = 
b       = 
v       = velocity (default value assigned to velocity in this rendition is -1)
angle   = The particular angle for which the microtubule is being calculated for 
(In later versions, the angle parameter has been altered to a container-data-structure filled with various angles for which microtubules are placed upon)

Additional note: when a = b, we label this as 'w'. (Not within this function explicitly however)
'''


# computing the underlying solution
def solve(rings, rays, r, d, t, q, p, a, b, v, tube_placements, t_ext_fact, output_specification):

    if len(tube_placements) > rays:
        raise IndexError(f'Too many microtubules requested: {len(tube_placements)}, within domain of {rays} angular rays.')

    for i in range(len(tube_placements)):
        if tube_placements[i] < 0 or tube_placements[i] > rays:
            raise IndexError(f'Angle {tube_placements[i]} is out of bounds, your range should be [0, {rays-1}]')

    d_radius = r / rings
    d_theta = ((2 * math.pi) / rays)
    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * d)

    time_steps = math.ceil(t / d_time)
    phi_center = 1 / (math.pi * (d_radius * d_radius))

    # For a unit test such that the initial particle density is NOT placed at the central patch
    # phi_center = 0

    # Used for displaying meta-data near the plot when tabulated data is visualized
    info_dict = {}

    if q:
        info_dict = pr.pr_quantities(rings, rays, r, d, t, d_radius, d_theta, d_time, time_steps, phi_center, a, b, v, tube_placements)
        print(f'Time Steps : {time_steps}')
        print(f'Approximated run time: {pr.convert_time((time_steps * rings * rays * 4 * 10 ** -6))}')

    # For a unit test such that the initial particle density is NOT placed at the central patch
    # phi_tab[0][0][0] = 1 / (d_radius * d_radius * d_theta)

    central_phi_tab = np.zeros([time_steps])
    mass_tab = np.zeros([time_steps])
    mass_loss_tab = np.zeros([time_steps])

    rho_mass_tab = np.zeros([time_steps])
    phi_mass_tab = np.zeros([time_steps])

    diffusive_layer = np.zeros([time_steps+1, rings, rays])
    advective_layer = np.zeros([time_steps+1, rings, rays])

    # Mean first passage time
    m_f_p_t = 0

# **** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    output_specification = np.unique(output_specification)

    global_max_flag = False

    for i in range(len(output_specification)):
        if output_specification[i] == 'global-max':
            global_max_flag = True

    k = 0
    m = 0
    n = 0

    while k < time_steps:

        net_current_out = 0

        if q and p >= 1:
            if k % p == 0:
                pr.pr_progress(k, time_steps)

        m = 0

        while m < rings:
            angle_index = 0
            if m == rings - 2:
                mass_loss_tab[k] = calc_loss_mass_j_r_r(diffusive_layer, k, d_radius, d_theta, rings, rays)
                if k > 0 and global_max_flag:
                    if mass_loss_tab[k-1] > mass_loss_tab[k]:
                        # glb_max_mass_loss = mass_loss_tab[k-1]
                        # acquires the global max from the mass function: mass versus time, used in computations for MFPT for varying W
                        glb_peak_time = (k-1) * d_time
                        print()
                        print(f'Global Peak Time: {glb_peak_time}')
                        glb_peak_time *= t_ext_fact
                        time_steps = math.ceil(glb_peak_time / d_time)

                        diffusive_layer = np.concatenate((diffusive_layer, np.zeros([diffusive_layer.shape[0], rings, rays])), axis=2)
                        advective_layer = np.concatenate((advective_layer, np.zeros([advective_layer.shape[0], rings, rays])), axis=2)
                        global_max_flag = False
            n = 0

            while n < rays:
                if m == rings - 1:
                    diffusive_layer[k+1][m][n] = 0
                else:
                    diffusive_layer[k+1][m][n] = u_density(diffusive_layer, k, m, n, d_radius, d_theta, d_time, phi_center, rings, advective_layer, angle_index, a, b, tube_placements)
                    if n == tube_placements[angle_index]:
                        # Update the associated tube within the dictionary
                        advective_layer[k+1][m][n] = u_tube(advective_layer, diffusive_layer, k, m, n, a, b, v, d_time, d_radius, d_theta)
                        if angle_index < len(tube_placements)-1:
                            angle_index = angle_index + 1
                if m == rings - 2:
                    net_current_out += j_r_r(diffusive_layer, k, m, n, d_radius, 0) * rings * d_radius * d_theta
                n += 1
            m += 1

        m_f_p_t += net_current_out * k * d_time * d_time

        central_phi_tab[k] = phi_center

        mass_tab[k] = calc_mass(diffusive_layer, advective_layer, k, d_radius, d_theta, phi_center, rings, rays, rho_mass_tab, phi_mass_tab, tube_placements)

        phi_center = u_center(diffusive_layer, k, d_radius, d_theta, d_time, phi_center, advective_layer, tube_placements, v)
        k += 1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    mass_loss_tab_ii = calc_loss_mass_derivative(mass_tab, d_time)
    print()

    # New implement feature added on 9/13/24 at 2:44 PM
    output_dictionary = {
        'mean-flow-passage-time': m_f_p_t,
        'diffusive_layer': diffusive_layer,
        'central_patch': central_phi_tab,
        'advective_layer': advective_layer,
        'total_mass': mass_tab,
        'mass_loss_rate_j_r_r': mass_tab,
        'mass_loss_rate_derivative': mass_loss_tab_ii
    }

    if q:
        print('Successful Completion.')

    return {key: value for key, value in output_dictionary.items() if key in [token.lower() for token in output_specification]}, info_dict

    # Implemented so that the user can decide exactly which data they would like to output for CSV or graphical purposes
    # an iterative return statement, ensures that the individual output-tokens are lower-cased via .lower() before comparison to output-dict tokens



