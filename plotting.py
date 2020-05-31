import numpy as np

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
    pos = [[p.x, p.y] for p in particles]
    pos = np.array(pos, dtype=np.float)

    weight = [p.w for p in particles]

    ax.scatter(pos[:, 0], pos[:, 1], marker='o', c=weight, s=2)


def plot_particles_grey(ax, particles):
    pos = [[p.x, p.y] for p in particles]
    pos = np.array(pos, dtype=np.float)

    ax.scatter(pos[:, 0], pos[:, 1], marker='o', color='grey', s=2)


def plot_association(ax, measurements, landmarks, assignment, color='purple'):
    N = len(measurements)

    for i in range(N):
        x, y = measurements[i]
        a, b = landmarks[assignment[i]]
        ax.plot([x, a], [y, b], color=color)