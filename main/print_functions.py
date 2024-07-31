import sys


def pr_quantities(rings, rays, r, d, t, d_radius, d_theta, d_time, time_steps, phi_center, a, b, v):
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
    print(f'a : {a}')
    print(f'b : {b}')
    print(f'Velocity: {v}')

    info_dict = {
        "Radial Curves": rings,
        "Angular Rays": rays,
        "Domain Radius": r,
        "Diffusion Coefficient": d,
        "Run Time": t,
        "Delta Radius": d_radius,
        "Delta Theta": d_theta,
        "Delta Time": d_time,
        "Time steps": time_steps,
        "Initial Central Density": phi_center,
        "a": a,
        "b": b,
        "Velocity": v
    }
    return info_dict


def pr_progress(c, o):
    percent = (c / o) * 100
    print(f"\rProgress: {int(percent+1)}%", end='', flush=True)

