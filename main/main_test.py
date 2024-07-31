# import calc
import calc_version_ii as calc
import tabulate_functions as tb
import plot_functions as plt
import numpy as np
import pandas as pd
import math
import os
from datetime import datetime
from playsound import playsound
from calc_version_iii_test import solve_ii


def tabulate_general(container, filepath, time, delta_time, title, rays):

    steps = [k * delta_time for k in range(1, len(container) + 1)]

    df = pd.DataFrame({'T': steps, 'Density': container})

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title}_general_tabulate_data{current_time}_time{time}_rays{rays}.csv"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    df.to_csv(os.path.join(filepath, filename), sep=',', index=False)


if __name__ == "__main__":

    rings = 25
    rays = 20
    # Diffusion coefficient
    d = 1
    # Domain Radius
    r = 1
    # Run time
    t = 1

    a = 0.1
    b = 0.1
    v = -1

    tube_placement = 1

    d_rad = r / rings
    d_the = ((2 * math.pi) / rays)
    # d_theta = 0.3306939635357677
    d_ti = (0.1 * min(d_rad * d_rad, d_the * d_the)) / (2 * d)
    time_steps = math.ceil(t / d_ti)

    test_filepath = "/Users/kbedoya88/Desktop/PROJECTS24/PyCharm/Theoretical-Biophysics/Bedoya-Kogan-Algorithm-Python/Algorithm-Version-VII/tabulate/test"

    comp = np.zeros([time_steps])
    d_phi = np.zeros([time_steps])
    d_rho = np.zeros([time_steps])
    d_center = np.zeros([time_steps])

    phi, central, mass, mass_loss, mass_loss_ii, d_time, d_radius, rho, info = solve_ii(rings, rays, r, d, t, True, 0, a, b, v, 1, comp, d_phi, d_rho, d_center)

    # tabulate_general(comp, test_filepath, t, d_time, "Component B", rays)
    # tabulate_general(d_phi, test_filepath, t, d_time, "Phi", rays)
    # tabulate_general(d_rho, test_filepath, t, d_time, "Rho", rays)
    # tabulate_general(d_center, test_filepath, t, d_time, "Center", rays)
