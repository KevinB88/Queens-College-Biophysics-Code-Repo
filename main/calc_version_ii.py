import math
import print_functions as pr
import numpy as np


# update density
def u_density(phi, rho, k, m, n, d_radius, d_theta, d_time, central, rings, angle, a, b):

    curr_density = phi[k][m][n]

    component_a = ((m+2) * j_r_r(phi, k, m, n, d_radius, rings)) - ((m+1) * j_l_r(phi, k, m, n, d_radius, central))

    component_a *= d_time / ((m+1) * d_radius)

    component_b = (j_r_t(phi, k, m, n, d_radius, d_theta)) - (j_l_t(phi, k, m, n, d_radius, d_theta))
    component_b *= d_time / ((m+1) * d_radius * d_theta)

    # if m == 0 and n == angle:
    #     print(f'time step: {k}, curve: {m}, ray: {n} Component b: {component_b}')

    if n == angle:  # if the current angle 'n' equates to the angle for which the microtubule is positioned at
        component_c = (a * phi[k][m][n]) * d_time - (b * rho[k][m]) * d_time
        # if m == 0:
        #     print(f' Component C at segment: {m} and at angle {n} : {component_c}')
    else:
        component_c = 0

    return curr_density - component_a - component_b - component_c
    # return curr_density - component_a - component_c
    # # return curr_density - component_a - component_b

    # if n == angle:
    #     return curr_density - component_a - component_b - (a * phi[k][m][n]) * d_time + (b * rho[k][m]) * d_time
    # else:
    #     return curr_density - component_a - component_b


def u_tube(rho, phi, k, m, n, a, b, v, rings, d_time, d_radius):

    if m == 0:
        prev_patch = 0
    else:
        prev_patch = rho[k][m-1]

    if m == rings - 1:
        curr_patch = 0
    else:
        curr_patch = rho[k][m+1]

    j_l = v * prev_patch
    j_r = v * curr_patch

    diff = ((j_r - j_l) / d_radius) * d_time

    return rho[k][m] - diff + (a * phi[k][m][n] * d_time) - (b * rho[k][m] * d_time), j_l


def u_tube_ii(rho, phi, k, m, n, a, b, v, d_time, d_radius):
    j_l = v * rho[k][m]
    if m == len(phi[k][m]) - 1:
        j_r = 0
    else:
        j_r = v * rho[k][m+1]

    return rho[k][m] - ((j_r - j_l) / d_radius) * d_time + a * phi[k][m][n] * d_time - b * rho[k][m] * d_time

    # if m == 0:
    #     prev = center
    # else:
    #     prev = phi[k][m-1][n]

    # return rho[k][m] - ((j_r - j_l) / d_radius) * d_time + a * prev * d_time - b * rho[k][m] * d_time


def u_tube_iii(rho, phi, k, m, n, a, b, v, d_time, d_radius):
    j_l = v * rho[k][m]
    if m == len(phi[k][m]) - 1:
        j_r = 0
    else:
        j_r = v * rho[k][m+1]

    return rho[k][m] - ((j_r - j_l) / d_radius) * d_time + a * phi[k][m][n] * d_time - b * rho[k][m] * d_time

# update central density
# def u_center(phi, k, d_radius, d_theta, d_time, curr_central):
#     total_sum = 0
#     for n in range(len(phi[k][0])):
#         total_sum += j_l_r(phi, k, 0, n, d_radius, curr_central)
#     total_sum *= (d_theta * d_time) / (math.pi * d_radius)
#     return curr_central - total_sum
#


# j_l = rho[k][m] * v
def u_center_ii(phi, m, k, d_radius, d_theta, d_time, curr_central, rho, v):
    total_sum = 0
    for n in range(len(phi[k][0])):
        total_sum += j_l_r(phi, k, 0, n, d_radius, curr_central)
    # d_theta = 0.3306939635357677
    total_sum *= (d_theta * d_time) / (math.pi * d_radius)
    a = curr_central - total_sum
    if m == 0:
        j_l = rho[k][m] * v
        return a + (abs(j_l) * d_time) / (math.pi * d_radius * d_radius)
    else:
        return a


# calculate for total mass
def calc_mass(phi, k, d_radius, d_theta, curr_central, rings, rays):
    mass = 0
    for m in range(rings):
        for n in range(rays):
            mass += phi[k][m][n] * (m+1)
    mass *= (d_radius * d_radius) * d_theta
    return (curr_central * math.pi * d_radius * d_radius) + mass


def calc_loss_mass_j(phi, k, d_radius, d_theta, rings, rays):
    total_sum = 0
    for n in range(rays):
        total_sum += j_r_r(phi, k, rings-2, n, d_radius, 0)
    total_sum *= rings * d_radius * d_theta

    return total_sum


# This should be calculated after the solution algorithm has successfully executed
# def calc_loss_mass_derivative(mass_container, d_time):
#     array = np.zeros([len(mass_container)-1])
#
#     for k in range(len(mass_container) - 1):
#         array[k] = (mass_container[k] - mass_container[k+1])/d_time
#
#     return array

def calc_loss_mass_derivative(mass_container, d_time):
    array = np.zeros([len(mass_container) - 1])
    for k in range(1, len(mass_container)):
        array[k-1] = (mass_container[k-1] - mass_container[k]) / d_time
    return array


# def calc_loss_mass_derivative(mass_container, d_time):
#     array = np.zeros([len(mass_container) - 1])
#     for k in range(1, len(mass_container) - 1):
#         array[k-1] = (mass_container[k-1] - mass_container[k+1]) / (2 * d_time)
#     return array


def calculate_total_mass(phi, k, radial_curves, angular_rays, center, d_radius, d_theta):
    total_sum = 0
    for m in range(radial_curves):
        for n in range(angular_rays):
            total_sum += phi[k][m] * (m + 1)
    total_sum *= (d_radius * d_radius) * d_theta
    return (center * math.pi * d_radius * d_radius) + total_sum


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
Selecting a specific to compute for 
'''

# def micro_tube_density(phi):
#

'''
rings   = Radial curves 
rays    = Angular rays
r       = Domain radius
d       = Diffusion coefficient
t       = Run time
q       = Passing a boolean value to determine if the user would like to print initial quantities
p       = Progression frequency for print operations
'''

# computing the underlying solution


# Updates the center dynamically

# Updates the center after all time step have come to an end
def solve_ii(rings, rays, r, d, t, q, p, a, b, v, angle):

    d_radius = r / rings
    d_theta = ((2 * math.pi) / rays)
    # d_theta = 0.3306939635357677
    # d_time = (0.05 * min(d_radius * d_radius, d_theta * d_theta)) / (2 * d)

    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * d)

    time_steps = math.ceil(t / d_time)
    phi_center = 1 / (math.pi * (d_radius * d_radius))
    # phi_center = 0

    info_dict = {}

    print(f"Delta Theta: {d_theta}")

    if q:
        info_dict = pr.pr_quantities(rings, rays, r, d, t, d_radius, d_theta, d_time, time_steps, phi_center, a, b, v)

    phi_tab = np.zeros([time_steps + 1, rings, rays])

    # phi_tab[0][0][0] = 1 / (d_radius * d_radius * d_theta)

    central_phi_tab = np.zeros([time_steps])
    mass_tab = np.zeros([time_steps])
    mass_loss_tab = np.zeros([time_steps])
    rho_tab = np.zeros([time_steps + 1, rings])

    for k in range(time_steps):
        if q and p >= 1:
            if k % p == 0:
                pr.pr_progress(k, time_steps)

        for m in range(rings):
            if m == rings - 2:
                mass_loss_tab[k] = calc_loss_mass_j(phi_tab, k, d_radius, d_theta, rings, rays)
            for n in range(rays):
                if m == rings - 1:
                    phi_tab[k+1][m][n] = 0
                else:
                    # def u_density(phi, rho, k, m, n, d_radius, d_theta, d_time, central, rings, angle, a, b):
                    phi_tab[k + 1][m][n] = u_density(phi_tab, rho_tab, k, m, n, d_radius, d_theta, d_time, phi_center, rings, angle, a, b)
                    if n == angle:
                        # rho_tab[k+1][m] = u_tube(rho_tab, phi_tab, k, m, n, a, b, v, rings, d_time, d_radius)
                        # def u_tube_iii(rho, phi, k, m, n, a, b, v, d_time, d_radius):
                        rho_tab[k+1][m] = u_tube_ii(rho_tab, phi_tab, k, m, n, a, b, v, d_time, d_radius)
                        # if m == 0 and k % 2 == 0 and k < 1500:
                        # if m == 0 and k % 2 == 0:
                        #     print(f"Value of rho at segment {m}, at time step: {k} : {rho_tab[k][m]} and center: {phi_center} and phi: {phi_tab[k][m][n]}")

        central_phi_tab[k] = phi_center

        mass_tab[k] = calc_mass(phi_tab, k, d_radius, d_theta, phi_center, rings, rays)

        # def u_center_ii(phi, m, k, d_radius, d_theta, d_time, curr_central, rho, v):
        # print(f'Time step {k} : Center : {phi_center}')
        phi_center = u_center_ii(phi_tab, 0, k, d_radius, d_theta, d_time, phi_center, rho_tab, v)

    mass_loss_tab_ii = calc_loss_mass_derivative(mass_tab, d_time)

    print()
    if q:
        print('Successful Completion.')
    return phi_tab, central_phi_tab, mass_tab, mass_loss_tab, mass_loss_tab_ii, d_time, d_radius, rho_tab, info_dict













