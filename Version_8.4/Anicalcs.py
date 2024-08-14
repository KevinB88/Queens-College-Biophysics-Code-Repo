import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation


# static polar-plane density plot
def static_pp_density(k, domain, center, rings, rays):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(f'Time step {k}')

    data = domain[k]
    theta = np.linspace(0, 2 * np.pi, rays)
    r = np.linspace(0, 1, rings)

    ax_radius, ax_theta = np.meshgrid(r, theta)
    z_ax = data.T

    cmap = cm.viridis

    c = ax.pcolormesh(ax_theta, ax_radius, z_ax, cmap=cmap)
    fig.colorbar(c, ax=ax)

    center_color = cmap(center[k])
    ax.plot(0, 0, 'o', color=center_color, markersize=10)


# animated polar-plane plot
def ani_pp_density(domain, time_steps, center, rings, rays):
    cmap = cm.viridis
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    def init():
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_title('Time Step 0')
        return []

    def update(k):
        ax.clear()
        ax.set_title(f'Time Step {k}')

        data = domain[k]

        theta = np.linspace(0, 2 * np.pi, rays)
        r = np.linspace(0, 1, rings)

        ax_r, ax_theta = np.meshgrid(r, theta)
        ax_z = data.T

        c = ax.pcolormesh(ax_theta, ax_r, ax_z, cmap=cmap, shading='auto')
        # fig.colorbar(c, ax=ax)

        center_color = cmap(center[k])
        ax.plot(0, 0, 'o', color=center_color, markersize=10)

        return []

    ani = FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=False, repeat=True)
    plt.show()
