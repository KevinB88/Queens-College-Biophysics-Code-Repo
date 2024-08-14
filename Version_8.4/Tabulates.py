import pandas as pd
import numpy as np
from datetime import datetime
import os


def tab_mfpt_by_w(mfpt_container, w_container, mt_count, filepath):
    df = pd.DataFrame({'W' : w_container, 'MFPT': mfpt_container})
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"MFPT_by_w_time{current_time}_MT_count{mt_count}.csv"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    df.to_csv(os.path.join(filepath, filename), sep=',', index=False)


def tab_rho_rad(container, filepath, time, delta_radius, w):

    final_state = container[len(container) - 1]
    radial_steps = [r * delta_radius for r in range(1, len(final_state) + 1)]

    df = pd.DataFrame({'R': radial_steps, 'D': final_state})
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rho_density_data_{current_time}_{time}_w={w}.csv"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    df.to_csv(os.path.join(filepath, filename), sep=',', index=False)


def tab_mass_base(container, filepath, time, delta_time, w):
    # time_steps = np.linspace(0, time, len(container))
    time_steps = [k * delta_time for k in range(1, len(container) + 1)]

    df = pd.DataFrame({'T': time_steps, 'M': container})

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mass_data_{current_time}_{time}_w={w}.csv"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    df.to_csv(os.path.join(filepath, filename), sep=',', index=False)


def tab_mass_loss(container, filepath, time, delta_time, w, calculation_type):

    # time_steps = np.linspace(0, time, len(container))
    time_steps = [k * delta_time for k in range(1, len(container) + 1)]

    df = pd.DataFrame({'T': time_steps, 'M': container})

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mass_loss_data_{current_time}_{time}_w{w}_{calculation_type}.csv"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    df.to_csv(os.path.join(filepath, filename), sep=',', index=False)


def tab_phi_ang(container, filepath, time, ring):
    angles = np.arange(0, len(container[0][0]), 1)
    density = container[len(container)-1][ring]
    df = pd.DataFrame({'Theta': angles, 'Density': density})
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"angle_density_{current_time}_time{time}.csv"
    df.to_csv(os.path.join(filepath, filename), sep=',', index=False)


def tab_phi_cent(container, filepath, time, delta_time):
    # time_steps = np.linspace(0, time, len(container))
    time_steps = [ k * delta_time for k in range(1, len(container) + 1) ]

    df = pd.DataFrame({'T': time_steps, 'M': container})
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"center_density_data_{current_time}_{time}.csv"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    df.to_csv(os.path.join(filepath, filename), sep=',', index=False)


def tab_phi_rad(container, filepath, angle, time, delta_radius, w):
    final_state = container[len(container)-1]

    # radial_steps = np.linspace(0, radius, len(container[0][0]))
    #
    radial_steps = [r * delta_radius for r in range(1, len(container[0][0])+1)]

    radial_densities = np.zeros([len(container[0][0])])

    for i in range(len(final_state)):
        radial_densities[i] = final_state[i][angle]

    df = pd.DataFrame({'R': radial_steps, 'D': radial_densities})

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"density_across_radius_data_date{current_time}_time{time}_angle{angle}_w={w}.csv"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    df.to_csv(os.path.join(filepath, filename), sep=',', index=False)


