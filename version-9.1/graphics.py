'''
Some colormaps/palettes to potentially use for the graphics

Viridis
Plasma
Inferno
Magma
Cividis
CoolWarm
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from datetime import datetime
from matplotlib.colors import LogNorm, BoundaryNorm
import os
import pandas as pd


# Both the color spectrum and the ticks are logarithmic
def polar_heat_map_log_log(diffusive_container, diffusive_central_container, radius_boundary, time_point, delta_time, toggle_border, w, v, MT_count, save_png, filepath):

    # Calculate the corresponding time step for an argument time point.
    time_step = math.ceil(time_point/delta_time)

    # Obtain a container for a given time step
    obtained_diffusive_container = diffusive_container[time_step]
    obtained_central_value = diffusive_central_container[time_step]

    data_center = np.full((1, len(obtained_diffusive_container)), obtained_central_value)
    obtained_diffusive_container = np.vstack([data_center, obtained_diffusive_container])

    print(f'Rings: {len(obtained_diffusive_container)}')
    print(f'Rays : {len(obtained_diffusive_container[0])}')

    r = np.linspace(0, radius_boundary, len(obtained_diffusive_container) + 1)
    theta = np.linspace(0, 2 * np.pi, len(obtained_diffusive_container[0]) + 1)

    R, Theta = np.meshgrid(r, theta)

    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    mx_density = np.max(obtained_diffusive_container)

    plt.figure(figsize=(8, 10))
    cmap = cm.get_cmap('viridis')

    if toggle_border:
        heatmap = plt.pcolormesh(X, Y, obtained_diffusive_container.T, shading='flat', cmap='viridis', norm=LogNorm(vmin=1e-6, vmax=mx_density), edgecolors='k', linewidth=0.2)
    else:
        heatmap = plt.pcolormesh(X, Y, obtained_diffusive_container.T, shading='flat', cmap='viridis', norm=LogNorm(vmin=1e-6, vmax=mx_density))

    # for i in range(1, len(r)-1):
    #     plt.text(X[i, 0], Y[i, 0], f'{i-1}', ha='center', va='center', color='white', fontsize=10, weight='bold')
    #
    # for j in range(0, len(theta), len(theta) // 8):
    #     plt.text(X[-1, j], Y[-1, j], f'{j * 360 // len(theta)}', ha='center', va='center', color='white', fontsize=10, weight='bold')

    cbar = plt.colorbar(heatmap, location='bottom', pad=0.08)
    cbar.ax.tick_params(labelsize=14, labelcolor='black')  # Increase tick size

    cbar_ticks = LogNorm(vmin=1e-6, vmax=mx_density).inverse(np.linspace(0, 1, num=8))
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{tick:.1e}' for tick in cbar_ticks ])
    plt.title(f'Time={time_point}, N={MT_count}, w={w}, v= {v}', fontdict={'weight': 'bold', 'font': 'Times New Roman', 'size': 20}, pad=20)

    plt.axis('off')

    if save_png:
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file = os.path.join(filepath, f'Time={time_point}_N={MT_count}_w={w}_Angles={len(obtained_diffusive_container[0])}_Rings={len(obtained_diffusive_container)}_data{current_time}.png')
            plt.savefig(file, bbox_inches='tight', transparent=True)
    plt.show()


# WE ARE USING THIS HEAT MAP
def polar_heat_map_linear_color_log(diffusive_container, diffusive_central_container, radius_boundary, time_point, delta_time, toggle_border, w, v, MT_count, save_png, filepath, color_scheme, show_plot=True):
    # Calculate the corresponding time step for an argument time point.
    time_step = math.ceil(time_point / delta_time)

    # Obtain a container for a given time step
    obtained_diffusive_container = diffusive_container[time_step]
    obtained_central_value = diffusive_central_container[time_step]

    data_center = np.full((1, len(obtained_diffusive_container)), obtained_central_value)
    obtained_diffusive_container = np.vstack([data_center, obtained_diffusive_container])

    print(f'Rings: {len(obtained_diffusive_container)}')
    print(f'Rays : {len(obtained_diffusive_container[0])}')

    r = np.linspace(0, radius_boundary, len(obtained_diffusive_container) + 1)
    theta = np.linspace(0, 2 * np.pi, len(obtained_diffusive_container[0]) + 1)

    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    # Set up a logarithmic color scale from 10^-7 to 10^0 (which is 1)
    # log_min = 1e-7
    # log_max = 1e0

    log_min = 10**-7
    log_max = 10**0

    plt.figure(figsize=(8, 10))
    cmap = cm.get_cmap(color_scheme, 512)

    boundaries = [0] + list(np.logspace(np.log10(log_min), np.log10(log_max), num=512))

    # Use BoundaryNorm with custom boundaries
    norm_zero = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    if toggle_border:
        heatmap = plt.pcolormesh(X, Y, obtained_diffusive_container.T, shading='flat', cmap=cmap, norm=norm_zero, edgecolors='k', linewidth=0.01)
    else:
        heatmap = plt.pcolormesh(X, Y, obtained_diffusive_container.T, shading='flat', cmap=cmap, norm=norm_zero)

    cbar = plt.colorbar(heatmap, location='bottom', pad=0.08)

    # Set ticks on a log scale between 10^-7 and 10^0, with 8 segments
    cbar_ticks = [0] + list(np.logspace(-7, 0, num=8)[1:])
    cbar.set_ticks(cbar_ticks)

    # Set tick labels as powers of ten
    cbar.set_ticklabels([f'0' if tick == 0 else f'$10^{{{int(np.log10(tick))}}}$' for tick in cbar_ticks ])
    cbar.ax.tick_params(labelsize=12, labelcolor='black')

    plt.title(f'Time={time_point}, N={MT_count}, w={w}, v= {v}', fontdict={'weight': 'bold', 'font': 'Times New Roman', 'size': 20}, pad=20)

    plt.axis('off')

    if save_png:
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file = os.path.join(filepath, f'Time={time_point}_N={MT_count}_w={w}_Angles={len(obtained_diffusive_container[0])}_Rings={len(obtained_diffusive_container)}_data{current_time}.png')
            plt.savefig(file, bbox_inches='tight', transparent=True)
    if show_plot:
        plt.show()


def polar_heat_map_linear_color_log_from_csv(csv_file, central_csv_file, radius_boundary, time_point, delta_time,
                                             toggle_border, w, v, MT_count, save_png, filepath, color_scheme,
                                             show_plot=True):
    # Load data from CSV
    diffusive_container = pd.read_csv(csv_file, header=None).values
    diffusive_central_container = pd.read_csv(central_csv_file, header=None).values

    # Calculate the corresponding time step for a given time point.
    time_step = math.ceil(time_point / delta_time)

    # Obtain a container for the given time step
    obtained_diffusive_container = diffusive_container
    obtained_central_value = diffusive_central_container[time_step-1]

    # Insert the central value into the diffusive container
    data_center = np.full((1, len(obtained_diffusive_container)), obtained_central_value)
    obtained_diffusive_container = np.vstack([data_center, obtained_diffusive_container ])

    print(f'Rings: {len(obtained_diffusive_container)}')
    print(f'Rays : {len(obtained_diffusive_container[0])}')

    # Set up polar coordinates for the heat map
    r = np.linspace(0, radius_boundary, len(obtained_diffusive_container) + 1)
    theta = np.linspace(0, 2 * np.pi, len(obtained_diffusive_container[ 0 ]) + 1)

    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    # Set up a logarithmic color scale
    log_min = 10 ** -7
    log_max = 10 ** 0

    plt.figure(figsize=(8, 10))
    cmap = cm.get_cmap(color_scheme, 512)

    # Define custom boundaries for color normalization
    boundaries = [0] + list(np.logspace(np.log10(log_min), np.log10(log_max), num=512))
    norm_zero = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    # Plot the heatmap
    if toggle_border:
        heatmap = plt.pcolormesh(X, Y, obtained_diffusive_container.T, shading='flat', cmap=cmap, norm=norm_zero,
                                 edgecolors='k', linewidth=0.01)
    else:
        heatmap = plt.pcolormesh(X, Y, obtained_diffusive_container.T, shading='flat', cmap=cmap, norm=norm_zero)

    # Add a colorbar
    cbar = plt.colorbar(heatmap, location='bottom', pad=0.08)

    # Set ticks on a log scale between 10^-7 and 10^0, with 8 segments
    cbar_ticks = [0] + list(np.logspace(-7, 0, num=8)[1:])
    cbar.set_ticks(cbar_ticks)

    # Set tick labels as powers of ten
    cbar.set_ticklabels([ f'0' if tick == 0 else f'$10^{{{int(np.log10(tick))}}}$' for tick in cbar_ticks ])
    cbar.ax.tick_params(labelsize=12, labelcolor='black')

    plt.title(f'Time={time_point}, N={MT_count}, w={w}, v= {v}',
              fontdict={'weight': 'bold', 'font': 'Times New Roman', 'size': 20}, pad=20)

    plt.axis('off')

    # Save the heatmap to a file if required
    if save_png:
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file = os.path.join(filepath,
                                f'Time={time_point}_N={MT_count}_w={w}_Angles={len(obtained_diffusive_container[ 0 ])}_Rings={len(obtained_diffusive_container)}_data{current_time}.png')
            plt.savefig(file, bbox_inches='tight', transparent=True)

    if show_plot:
        plt.show()


def polar_heat_map_linear(diffusive_container, diffusive_central_container, radius_boundary, time_point, delta_time, toggle_border, w, v, MT_count, save_png, filepath, color_scheme):

    # Calculate the corresponding time step for an argument time point.
    time_step = math.ceil(time_point/delta_time)

    # Obtain a container for a given time step
    obtained_diffusive_container = diffusive_container[time_step]
    obtained_central_value = diffusive_central_container[time_step]

    data_center = np.full((1, len(obtained_diffusive_container)), obtained_central_value)
    obtained_diffusive_container = np.vstack([data_center, obtained_diffusive_container])

    print(f'Rings: {len(obtained_diffusive_container)}')
    print(f'Rays : {len(obtained_diffusive_container[0])}')

    r = np.linspace(0, radius_boundary, len(obtained_diffusive_container) + 1)
    theta = np.linspace(0, 2 * np.pi, len(obtained_diffusive_container[0]) + 1)

    R, Theta = np.meshgrid(r, theta)

    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    mx_density = np.max(obtained_diffusive_container)

    plt.figure(figsize=(8, 10))
    cmap = cm.get_cmap(color_scheme)

    if toggle_border:
        heatmap = plt.pcolormesh(X, Y, obtained_diffusive_container.T, shading='flat', cmap=cmap, vmin=0, vmax=mx_density, edgecolors='k', linewidth=0.01)
    else:
        heatmap = plt.pcolormesh(X, Y, obtained_diffusive_container.T, shading='flat', cmap=cmap, vmin=0, vmax=mx_density)

    # for i in range(1, len(r)-1):
    #     plt.text(X[i, 0], Y[i, 0], f'{i-1}', ha='center', va='center', color='white', fontsize=10, weight='bold')
    #
    # for j in range(0, len(theta), len(theta) // 8):
    #     plt.text(X[-1, j], Y[-1, j], f'{j * 360 // len(theta)}', ha='center', va='center', color='white', fontsize=10, weight='bold')

    cbar = plt.colorbar(heatmap, location='bottom', pad=0.08)
    cbar_ticks = np.linspace(0, mx_density, num=6)
    cbar.set_ticks(cbar_ticks)

    if mx_density < 1 * 10 ** -6:
        cbar.set_ticklabels([f'{np.round(tick, 10)}' for tick in cbar_ticks])
    else:
        cbar.set_ticklabels([f'{np.round(tick, 6)}' for tick in cbar_ticks])
    cbar.ax.tick_params(labelsize=12, labelcolor='black')

    plt.title(f'Time={time_point}, N={MT_count}, w={w}, v= {v}', fontdict={'weight': 'bold', 'font': 'Times New Roman', 'size': 20}, pad=20)

    plt.axis('off')

    if save_png:
        if filepath:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file = os.path.join(filepath, f'Time={time_point}_N={MT_count}_w={w}_Angles={len(obtained_diffusive_container[0])}_Rings={len(obtained_diffusive_container)}_data{current_time}.png')
            plt.savefig(file, bbox_inches='tight', transparent=True)
    plt.show()




