from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_animation_matplotlib(data, center_data, time_steps, radial_bins, angular_bins):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    theta = np.linspace(0, 2 * np.pi, angular_bins)
    r = np.linspace(0, 1, radial_bins)
    R, Theta = np.meshgrid(r, theta)

    norm_center_data = (center_data - np.min(center_data)) / (np.max(center_data) - np.min(center_data))

    Z = data[0]
    contours = ax.contourf(Theta, R, Z.T, cmap='viridis')
    scatter = ax.scatter(0, 0, s=100, c=plt.cm.viridis(norm_center_data[0]), cmap='viridis', edgecolors='k')
    colorbar = plt.colorbar(contours, ax=ax, orientation='vertical', pad=0.1, aspect=30)

    def update(frame):
        for c in ax.collections:
            c.remove()
        Z = data[frame]
        contours = ax.contourf(Theta, R, Z.T, cmap='viridis')
        scatter.set_color(plt.cm.viridis(norm_center_data[frame]))
        return contours.collections + [scatter]

    ani = animation.FuncAnimation(fig, update, frames=range(time_steps), blit=False, repeat=False)
    plt.show()


def create_animation(data, center_data, time_steps, radial_bins, angular_bins):

    def update_plot(t):
        mlab.clf()
        theta = np.linspace(0, 2 * np.pi, angular_bins)
        r = np.linspace(0, 1, radial_bins)
        R, Theta = np.meshgrid(r, theta)
        Z = data[t]
        mlab.surf(R * np.cos(Theta), R * np.sin(Theta), Z)
        mlab.points3d(0, 0, center_data[t], scale_factor=0.1)
        mlab.draw()

    for t in range(time_steps):
        update_plot(t)
        mlab.process_ui_events()

    mlab.show()





