import math
import Prints as pr
import numpy as np

# !This version only calculates the microtubule across a single angle!


# Update contents at the diffusive layer (Particle density computation across the domain), returns a floating point number
def u_density(phi, rho, k, m, n, d_radius, d_theta, d_time, central, rings, angle, a, b):

    curr_density = phi[k][m][n]

    component_a = ((m+2) * j_r_r(phi, k, m, n, d_radius, rings)) - ((m+1) * j_l_r(phi, k, m, n, d_radius, central))

    component_a *= d_time / ((m+1) * d_radius)

    component_b = (j_r_t(phi, k, m, n, d_radius, d_theta)) - (j_l_t(phi, k, m, n, d_radius, d_theta))
    component_b *= d_time / ((m+1) * d_radius * d_theta)

    if n == angle:  # if the current angle 'n' equates to the angle for which the microtubule is positioned at
        component_c = (a * phi[k][m][n]) * d_time - (((b * rho[k][m]) * d_time) / ((m+1) * d_radius * d_theta))
        # if m == 0:
        #     print(f' Component C at segment: {m} and at angle {n} : {component_c}')
    else:
        component_c = 0

    return curr_density - component_a - component_b - component_c


# Update contents at advective layer (Particle density computation along a microtubule), returns a floating point number
def u_tube(rho, phi, k, m, n, a, b, v, d_time, d_radius, d_theta):
    j_l = v * rho[k][m]
    if m == len(phi[k][m]) - 1:
        j_r = 0
    else:
        j_r = v * rho[k][m+1]

    return rho[k][m] - ((j_r - j_l) / d_radius) * d_time + (a * phi[k][m][n] * (m+1) * d_radius * d_theta) * d_time - b * rho[k][m] * d_time


# Update the central patch, returns a floating point value
def u_center(phi, k, d_radius, d_theta, d_time, curr_central, rho, v):
    total_sum = 0
    for n in range(len(phi[k][0])):
        total_sum += j_l_r(phi, k, 0, n, d_radius, curr_central)
    # d_theta = 0.3306939635357677
    total_sum *= (d_theta * d_time) / (math.pi * d_radius)
    a = curr_central - total_sum

    j_l = rho[k][0] * v
    return a + (abs(j_l) * d_time) / (math.pi * d_radius * d_radius)


# calculate for total mass across domain, returns a floating point number
def calc_mass(phi, rho, k, d_radius, d_theta, curr_central, rings, rays):
    mass = 0
    for m in range(rings):
        for n in range(rays):
            mass += phi[k][m][n] * (m+1)
    mass *= (d_radius * d_radius) * d_theta

    # global_diffusive_mass[k] = mass + curr_central * math.pi * d_radius * d_radius

    microtubule_mass = 0
    for m in range(len(rho[k])):
        microtubule_mass += rho[k][m] * d_radius

    return (curr_central * math.pi * d_radius * d_radius) + mass + microtubule_mass


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
def solve(rings, rays, r, d, t, q, p, a, b, v, angle):

    d_radius = r / rings
    d_theta = ((2 * math.pi) / rays)
    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * d)

    time_steps = math.ceil(t / d_time)
    phi_center = 1 / (math.pi * (d_radius * d_radius))

    # For a unit test such that the initial particle density is NOT placed at the central patch
    # phi_center = 0

    # Used for displaying meta-data near the plot when tabulated data is visualized
    info_dict = {}

    print(f"Delta Theta: {d_theta}")

    if q:
        info_dict = pr.pr_quantities(rings, rays, r, d, t, d_radius, d_theta, d_time, time_steps, phi_center, a, b, v, [angle])

    phi_tab = np.zeros([time_steps + 1, rings, rays])

    # For a unit test such that the initial particle density is NOT placed at the central patch
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
                mass_loss_tab[k] = calc_loss_mass_j_r_r(phi_tab, k, d_radius, d_theta, rings, rays)
            for n in range(rays):
                if m == rings - 1:
                    phi_tab[k+1][m][n] = 0
                else:

                    phi_tab[k + 1][m][n] = u_density(phi_tab, rho_tab, k, m, n, d_radius, d_theta, d_time, phi_center, rings, angle, a, b)
                    if n == angle:

                        rho_tab[k+1][m] = u_tube(rho_tab, phi_tab, k, m, n, a, b, v, d_time, d_radius, d_theta)

        central_phi_tab[k] = phi_center

        mass_tab[k] = calc_mass(phi_tab, rho_tab, k, d_radius, d_theta, phi_center, rings, rays)

        phi_center = u_center(phi_tab, k, d_radius, d_theta, d_time, phi_center, rho_tab, v)

    mass_loss_tab_ii = calc_loss_mass_derivative(mass_tab, d_time)

    print()
    if q:
        print('Successful Completion.')
    return phi_tab, central_phi_tab, mass_tab, mass_loss_tab, mass_loss_tab_ii, d_time, d_radius, rho_tab, info_dict











