import os.path
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import datetime
import tkinter as tk
import mplcursors
from tkinter import messagebox
import math
import pandas as pd


# The rings array specifies which rings the user would like to observe

def plot_multiple_general(container, key):

    plt.figure(figsize=(10, 6))

    for i in range(len(container)):
        df = pd.read_csv(container[i])
        plt.plot(df['Theta'], df['Density'], label=f'{key[i]}')

    plt.ylabel('Phi')
    plt.xlabel('Theta')
    plt.legend()
    plt.show()


def plot_phi_angle_multiple_rings(phi, rings, time, discrete, limit, text, filepath, save_png):

    if len(rings) > len(phi[0]) or len(rings) == 0:
        print("Out of bounds!")
        print(f"Phi currently has {len(phi[0])} rings, your input array has: {len(rings)} rings.")
        return

    x = np.arange(0, len(phi[0][0]), 1)

    global_maxes = np.zeros([len(rings)])
    global_mins = np.zeros([len(rings)])

    for i in range(len(rings)):
        if rings[i] > len(phi[0])-1 or rings[i] < 0:
            continue
        else:
            ring = rings[i]
            global_maxes[i] = np.max(phi[len(phi)-1][ring])
            # print(np.max(phi[len(phi)-1][ring]))
            global_mins[i] = np.min(phi[len(phi)-1][ring])

    # global_max = np.max(global_maxes)
    # global_min = np.min(global_mins)

    global_max = 0.9
    global_min = 0.7

    plt.figure(figsize=(12, 8))

    max = -1
    min = len(phi[0])

    for m in range(len(rings)):
        if rings[m] > len(phi[0]) - 1 or rings[m] < 0:
            continue
        else:
            if rings[m] > max:
                max = rings[m]
            if rings[m] < min:
                min = rings[m]
            # y = phi[len(phi)-1][rings[m]]
            ring = rings[m]
            y = phi[len(phi)-1][ring]
            # y = phi[len(phi)-1, ring, :]
        if discrete:
            plt.scatter(x, y, label=f'Ring: {rings[m]}')
        else:
            plt.plot(x, y, label=f'Ring: {rings[m]}')

    plt.title(f'Phi along Angular Rays(N) at Radial Curves(M) : [{min}, {max}], at time: {time}')
    plt.xlabel('Theta (N)')
    plt.ylabel('Phi')
    if limit:
        plt.ylim(global_min, global_max)
    legend = plt.legend()
    legend.set_draggable(True)

    plt.xticks(np.arange(0, len(phi[0][0]), 1))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=False, prune='both'))

    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.0f}, {sel.target[1]:.5f})"))
    info_text = "\n".join([f"{key}: {value}" for key, value in text.items()])
    plt.annotate(info_text, xy=(1.05, 0.5), xycoords='axes fraction', fontsize=10, verticalalignment='center', bbox=dict(boxstyle="round, pad=0.3", edgecolor='black', facecolor='white'))

    plt.subplots_adjust(right=0.8)
    plt.tight_layout()

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'phi_angle_time{time}_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {filepath}')
    plt.show()


def plot_phi_angle(container, ring, time, discrete):

    if ring > len(container[0])-1 or ring < 0:
        print(f"Your ring must fall between: [0, {len(container[0])-1}]")
        return

    x = np.arange(0, len(container[0][0]), 1)
    y = container[len(container)-1][ring]

    if discrete:
        plt.scatter(x, y, color='blue')
    else:
        plt.plot(x, y, color='blue')

    plt.title(f'Phi at ring: {ring}, along angles of Theta, at time: {time}')
    plt.xlabel('Theta')
    plt.ylabel('Phi')

    # displaying discrete points along an axis.
    plt.xticks(np.arange(0, len(container[0][0]), 1))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=False, prune='both'))

    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.0f}, {sel.target[1]:.5f})"))

    plt.show()

def plot_rho_density(container, radius, time, text, filepath, save_png, discrete):

    rho_final_step = container[len(container)-1]
    x = np.linspace(0, radius, len(container[0]))
    y = rho_final_step

    plt.figure(figsize=(12, 8))
    if discrete:
        plt.scatter(x, y)
    else:
        plt.plot(x, y, 'blue')

    plt.xlabel('Time')
    plt.ylabel('Rho Density')

    info_text = "\n".join([ f"{key}: {value}" for key, value in text.items() ])
    plt.annotate(info_text, xy=(1.05, 0.5), xycoords='axes fraction', fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle="round, pad=0.3", edgecolor='black', facecolor='white'))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=False, prune='both'))

    plt.subplots_adjust(right=0.8)
    plt.tight_layout()

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'rho_radius_plot_time{time}_data{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {filepath}')

    plt.show()


def plot_mass_loss(container, time):
    x = np.linspace(0, time, len(container))
    y = [container[k] for k in range(len(container))]

    plt.plot(x, y, 'blue')
    plt.xlabel('Time')
    plt.ylabel('Mass loss')
    plt.show()


def plot_moss_loss_comparison(container_i, container_ii, time):
    x_i = np.linspace(0, time, len(container_i))
    y_i = [container_i[k] for k in range(len(container_i))]

    x_ii = np.linspace(0, time, len(container_ii))
    y_ii = [container_ii[k] for k in range(len(container_ii))]

    plt.plot(x_i, y_i, linewidth=3, color='blue')
    plt.plot(x_ii, y_ii, linewidth=1, color='red')

    plt.xlabel('time')
    plt.ylabel('mass loss')
    plt.show()


def plot_mass(mass_container, time):
    x = np.linspace(0, time, len(mass_container))
    y = [mass_container[k] for k in range(len(mass_container))]
    plt.plot(x, y, 'blue')
    plt.xlabel('Time')
    plt.ylabel('Total mass')
    plt.show()


def plot_central_density(central_container, time):
    x = np.linspace(0, time, len(central_container))
    y = [central_container[k] for k in range(len(central_container))]
    plt.plot(x, y, 'black')
    plt.xlabel('Time')
    plt.ylabel('Central Density')
    plt.show()


def plot_all_densities(container, radius, angle, freq):
    x = np.linspace(0, radius, len(container[0][0]))
    for k in range(len(container)):
        if k % freq == 0:
            radial_densities = np.zeros([len(container[0][0])])
            for i in range(len(container[0][0])):
                radial_densities[i] = container[k][i][angle]
            y = radial_densities

            plt.plot(x, y, 'blue')

    plt.xlabel('Radius')
    plt.ylabel('Density')
    plt.show()


# Additional feature : within the plot, specify which angle is being observed within the domain
def plot_final_density(container, radius, angle):
    final_state = container[len(container) - 1]
    radial_densities = np.zeros([len(container[0][0])])

    for i in range(len(final_state)):
        radial_densities[i] = final_state[i][angle]

    x = np.linspace(0, radius, len(container[0][0]))
    # y = [radial_densities[m] for m in range(len(radial_densities))]
    y = radial_densities

    plt.plot(x, y, 'black')
    plt.xlabel('Radius')
    plt.ylabel('Density')
    plt.show()


# assuming that the initial run-time is 1, and there are 49,999 time steps
def plot_density_heat_map(density, central, desired_time):

    time_step = math.floor(len(density) * desired_time)

    phi = density[time_step]
    c_phi = central[time_step]

    radial_values = np.arange(0, len(density[0]) + 1)
    angular_values = np.linspace(0, 2 * np.pi, len(density[0][0]))

    phi_with_center = np.zeros((len(density[0] + 1), len(density[0][0])))
    phi_with_center[0, :] = c_phi
    phi_with_center[1:, :] = phi

    norm = Normalize(vmin=phi_with_center.min(), vmax=phi_with_center.max())
    phi_normalized = norm(phi_with_center)

    r, theta = np.meshgrid(radial_values, angular_values)

    plt.figure()
    ax = plt.subplot(projection='polar')

    c = ax.pcolormesh(theta, r, shading='auto', cmap='viridis')
    # c = ax.pcolormesh(theta, r, phi_normalized.T, shading='auto', cmap='viridis')

    plt.colorbar(c, label='Normalized Density')

    plt.title(f'Density Distribution at time step {time_step}')
    plt.xlabel('Angle')
    plt.ylabel('Radius')
    plt.show()


def plot_density_heat_map_discrete(density, central, time_step):

    phi = density[time_step]
    c_phi = central[time_step]
    # c_phi = 0

    radial_values = np.arange(0, len(density[0]) + 1)
    angular_values = np.linspace(0, 2 * np.pi, len(density[0][0]))

    phi_with_center = np.zeros((len(density[0]) + 1, len(density[0][0])))
    phi_with_center[0, :] = c_phi
    phi_with_center[1:, :] = phi

    norm = Normalize(vmin=phi_with_center.min(), vmax=phi_with_center.max())
    phi_normalized = norm(phi_with_center)

    r, theta = np.meshgrid(radial_values, angular_values)

    plt.figure()
    ax = plt.subplot(projection='polar')

    c = ax.pcolormesh(theta, r, phi_normalized.T, shading='auto', cmap='viridis')

    plt.colorbar(c, label='Normalized Density Legend')

    plt.title(f'Density Distribution at time step {time_step}')
    plt.xlabel('Angle')
    plt.ylabel('Radius')
    plt.show()


def plot_density_heat_map_discrete_animated(density, central, time_step_bound):
    num_radial_curves = len(density[0])
    num_angles = len(density[0][0])

    radial_values = np.linspace(0, 1, num_radial_curves + 1)
    angular_values = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    r, theta = np.meshgrid(radial_values, angular_values)
    phi_with_center = np.zeros((num_angles, num_radial_curves + 1))
    c = ax.pcolormesh(theta, r, phi_with_center, shading='auto', cmap='viridis')

    plt.colorbar(c, ax=ax, label='Normalized Density')

    # Set radial ticks
    ax.set_rticks(radial_values)
    ax.set_rlabel_position(-22.5)

    # Set angular ticks
    ax.set_xticks(np.linspace(0, 2 * np.pi, num_angles, endpoint=False))

    ax.grid(True)

    norm = Normalize(vmin=density.min(), vmax=density.max())  # Normalize over the entire dataset

    def update(frame):
        phi = density[frame]
        c_phi = central[frame]

        phi_with_center[:, 0] = c_phi
        phi_with_center[:, 1:] = phi.T

        c.set_array(norm(phi_with_center).ravel())
        return c,

    ani = FuncAnimation(fig, update, frames=range(time_step_bound), blit=True, repeat=False)
    plt.show()