import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from particle import FlatParticle

def plot_history(ax, history, color='green'):
    for x, y, _ in history:
        ax.plot([x], [y], marker='o', markersize=3, color=color)

    if len(history) > 1:
        for i in range(len(history) - 1):
            x, y, _ = history[i]
            a, b, _ = history[i + 1]
            ax.plot([x, a], [y, b], color=color)
     

def plot_landmarks(ax, landmarks, color='blue'):
    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color=color)


def plot_measurement(ax, pos, landmarks, color):
    landmarks = landmarks + pos
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=20, marker='o', color=color)


def plot_connections(ax, s, landmarks, color='purple'):
    x, y, _ = s
    for a, b in landmarks:
        ax.plot([x, a], [y, b], color=color)


def plot_particles_weight(ax, particles):
    pos = np.zeros((FlatParticle.len(particles), 2), dtype=np.float32)
    pos[:, 0] = FlatParticle.x(particles)
    pos[:, 1] = FlatParticle.y(particles)

    weight = FlatParticle.w(particles)

    ax.scatter(pos[:, 0], pos[:, 1], marker='o', c=weight, s=2)


def plot_particles_grey(ax, particles):
    pos = np.zeros((FlatParticle.len(particles), 2), dtype=np.float32)
    pos[:, 0] = FlatParticle.x(particles)
    pos[:, 1] = FlatParticle.y(particles)

    ax.scatter(pos[:, 0], pos[:, 1], marker='o', color='grey', s=2)


def plot_association(ax, measurements, landmarks, assignment, color='purple'):
    N = len(measurements)

    for i in range(N):
        x, y = measurements[i]
        a, b = landmarks[assignment[i]]
        ax.plot([x, a], [y, b], color=color)


def plot_confidence_ellipse(ax, landmark, cov, n_std=1.0, edgecolor='red'):
    x, y = landmark
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, edgecolor=edgecolor, facecolor='none')

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)