import os.path
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from datetime import datetime
import tkinter as tk
import mplcursors
from tkinter import messagebox
import math
import pandas as pd
import re
import string
from PIL import Image

# The rings array specifies which rings the user would like to observe


def plt_mean_passage(w_x_container, m_y_container, save_path, save_png, show_plot, plt_type):

    x = w_x_container
    y = m_y_container

    if plt_type.lower() == 'cont':
        plt.plot(x, y)
    elif plt_type.lower() == 'disc':
        plt.scatter(x, y)
    elif plt_type.lower() == 'log':
        plt.loglog(x, y)

    plt.xlabel('W')
    plt.ylabel('Mean First Passage Time')
    plt.title('Mean first passage time versus w')

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file = os.path.join(save_path, f'MFPT_data_{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {save_path}')
    if show_plot:
        plt.show()


def plot_array(container, x_lbl, y_lbl, delta_time, title):

    time_steps = [k * delta_time for k in range(1, len(container) + 1)]

    df = pd.DataFrame({f'{x_lbl}': time_steps, f'{y_lbl}': container})

    plt.title(title)
    plt.plot(df[f'{x_lbl}'], df[f'{y_lbl}'])
    plt.xlabel(f'{x_lbl}')
    plt.ylabel(f'{y_lbl}')
    plt.show()


def plot_generic_csv(csv_file, x_lbl, y_lbl):
    df = pd.read_csv(csv_file)
    plt.plot(df[f'{x_lbl}'], df[f'{y_lbl}'])
    plt.xlabel(f'{x_lbl}')
    plt.ylabel(f'{y_lbl}')
    plt.show()


def extract_key(filename):
    pattern = r'_0\.1_w=([-+]?[0-9]*\.?[0-9]+)\.csv'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def plot_csv_files(directory, x_lbl, y_lbl, title, save_png, show_plot, save_path):
    plt.figure(figsize=(10,6))

    files_with_keys = []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            key = extract_key(filename)
            if key is not None:
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                files_with_keys.append((key, df))
    files_with_keys.sort(key=lambda x: x[0])

    for key, df in files_with_keys:
        plt.plot(df[x_lbl], df[y_lbl], label=f'w={key}')
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    plt.legend()

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file = os.path.join(save_path, f'{title}_data_{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {save_path}')
    if show_plot:
        plt.show()


def plot_multiple_general(container, key, save_png, filepath, show_plot, x_lbl, y_lbl, x_title, y_title, title):

    plt.figure(figsize=(10, 6))

    for i in range(len(container)):
        df = pd.read_csv(container[i])

        plt.scatter(df[f'{x_lbl}'], df[f'{y_lbl}'], label=f'{key[i]}', linewidth=(10/(i+1)))
        plt.yscale('log')
        plt.xscale('log')

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'plots_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {filepath}')
    plt.show()
    if show_plot:
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


def plt_phi_ang(container, ring, time, discrete, save_png, filepath, plot_show):

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

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'density_angular_ring_{ring}_time_{time}_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
    if plot_show:
        plt.show()

'''

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'base_mass_time_{time}_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
    if plot_show:
        plt.show()

'''


def plt_rho_rad(container, radius, time, text, filepath, save_png, discrete, plot_show, angle):

    rho_final_step = container[len(container)-1]
    x = np.linspace(0, radius, len(container[0]))
    y = rho_final_step

    plt.figure(figsize=(12, 8))
    if discrete:
        plt.scatter(x, y)
    else:
        plt.plot(x, y, 'blue')

    plt.xlabel('Radius')
    plt.ylabel('Rho Density')

    info_text = "\n".join([ f"{key}: {value}" for key, value in text.items() ])
    plt.annotate(info_text, xy=(1.05, 0.5), xycoords='axes fraction', fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle="round, pad=0.3", edgecolor='black', facecolor='white'))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=False, prune='both'))

    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.title(f'Density across radius of MT at angle: {angle}')

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'rho_radius_plot_time{time}_data{current_time}_angle{angle}.png')
            plt.savefig(file, bbox_inches='tight')
            print(f'Plot saved to {filepath}')
    if plot_show:
        plt.show()


def plt_mass_loss(container, time, save_png, filepath, plot_show):
    x = np.linspace(0, time, len(container))
    y = [container[k] for k in range(len(container))]

    plt.title(f'Mass loss rate versus time : {time}')
    plt.plot(x, y, 'blue')
    plt.xlabel('Time')
    plt.ylabel('Mass loss')

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'mass_loss_time_{time}_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
    if plot_show:
        plt.show()


def plot_moss_loss_comparison(container_i, container_ii, time, plot_show):
    x_i = np.linspace(0, time, len(container_i))
    y_i = [container_i[k] for k in range(len(container_i))]

    x_ii = np.linspace(0, time, len(container_ii))
    y_ii = [container_ii[k] for k in range(len(container_ii))]

    plt.plot(x_i, y_i, linewidth=3, color='blue')
    plt.plot(x_ii, y_ii, linewidth=1, color='red')

    plt.xlabel('time')
    plt.ylabel('mass loss')
    if plot_show:
        plt.show()


def plt_mass_base(mass_container, time, save_png, filepath, plot_show, title):
    x = np.linspace(0, time, len(mass_container))
    # y = [mass_container[k] for k in range(len(mass_container))]
    y = mass_container
    plt.title(f'Mass within domain versus time : {time}')
    plt.plot(x, y, 'blue')
    plt.xlabel('Time')
    plt.ylabel('Total mass')
    plt.title(title)
    plt.ylim(.99, 1.01)

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'base_mass_time_{time}_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
    if plot_show:
        plt.show()


def plt_phi_center(central_container, time, save_png, filepath, plot_show):
    x = np.linspace(0, time, len(central_container))
    y = [central_container[k] for k in range(len(central_container))]
    plt.title(f'Central Densities Plot at time {time}')
    plt.plot(x, y, 'black')
    plt.xlabel('Time')
    plt.ylabel('Central Density')
    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'central_density_time_{time}_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight')
    if plot_show:
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
def plt_phi_rad(container, radius, angle, save_png, filepath, time, plot_show, d_time, time_point):

    # Acquiring the corresponding time-step:
    time_stamp = math.ceil(time_point / d_time)

    final_state = container[time_stamp]
    radial_densities = np.zeros([len(container[0][0])])

    for i in range(len(final_state)):
        radial_densities[i] = final_state[i][angle]

    x = np.linspace(0, radius, len(container[0][0]))
    # y = [radial_densities[m] for m in range(len(radial_densities))]
    y = radial_densities

    plt.title(f'Phi versus radius at angle: {angle} at time t={time_point}')
    plt.plot(x, y, 'black')
    plt.xlabel('Radius')
    plt.ylabel('Density')

    if save_png:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            file = os.path.join(filepath, f'final_density_angle:{angle}_time_{time}_date{current_time}.png')
            plt.savefig(file, bbox_inches='tight')

    if plot_show:
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


def plot_density_heat_map_discrete(density, central, time_step, show_plot):

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

    if show_plot:
        plt.show()


def plot_density_heat_map_discrete_animated(density, central, time_step_bound, file_path=''):
    # Generate a default filename based on the current time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_filename = f'diffusion_animation_{current_time}.mp4'

    # Use the provided file path or the default filename
    if file_path:
        # Ensure the directory exists
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        filename = os.path.join(file_path, default_filename)
    else:
        filename = default_filename

    num_radial_curves = len(density[0])
    num_angles = len(density[0][0])

    radial_values = np.linspace(0, 1, num_radial_curves + 1)
    angular_values = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    r, theta = np.meshgrid(radial_values, angular_values)
    phi_with_center = np.zeros((num_angles, num_radial_curves + 1))
    c = ax.pcolormesh(theta, r, phi_with_center, shading='auto', cmap='plasma')

    plt.colorbar(c, ax=ax, label='Normalized Density')

    # Set radial ticks
    ax.set_rticks(radial_values)
    ax.set_rlabel_position(-22.5)

    # Set angular ticks
    ax.set_xticks(np.linspace(0, 2 * np.pi, num_angles, endpoint=False))

    ax.grid(True)

    # Define a suitable maximum value for normalization
    max_value = max(density.max(), central.max())
    norm = Normalize(vmin=0, vmax=max_value)  # Normalize between 0 and max_value

    def update(frame):
        phi = density[frame]
        c_phi = central[frame]

        phi_with_center[:, 0] = c_phi
        phi_with_center[:, 1:] = phi.T

        c.set_array(norm(phi_with_center).ravel())
        return c,

    ani = FuncAnimation(fig, update, frames=range(time_step_bound), blit=True, repeat=False)

    # Save animation
    writer = FFMpegWriter(fps=10)
    try:
        ani.save(filename, writer=writer)
        print(f"Animation successfully saved as {filename}")
    except Exception as e:
        print(f"An error occurred while saving the animation: {e}")

    plt.show()