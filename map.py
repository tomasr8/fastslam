import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from plotting import plot_landmarks
from particle import FlatParticle

def kmeans(X, weights, initial_centroids, max_iterations=10):
    start = time.time()

    k = initial_centroids.shape[0]
    centroids = initial_centroids

    dist = distance.cdist(X, centroids, "euclidean")
    dist = dist * weights
    assignment = np.argmin(dist, axis=1)

    for _ in range(max_iterations):
        centroids = np.vstack(
            [
                (np.sum(X[assignment == i, :] * weights[assignment == i], axis=0) / np.sum(weights[assignment == i])) for i in range(k)
            ]
        )

        dist = distance.cdist(X, centroids, "euclidean")
        dist = dist * weights
        next_assignment = np.argmin(dist, axis=1)

        if np.array_equal(assignment, next_assignment):
            break

        assignment = next_assignment

    print("kmeans inside: ", time.time() - start)

    return assignment, centroids


def compute_map(particles, n=100):
    start = time.time()
    X = []
    weights = []

    idx = np.argsort(-FlatParticle.w(particles))[:n]
    for i in idx:
        landmarks = FlatParticle.get_landmarks(particles, i)
        p = FlatParticle.get_particle(particles, i)
        for landmark in landmarks:
            X.append(landmark)
            weights.append(p[3])

    X = np.array(X, dtype=np.float32)
    size = len(weights)
    weights = np.array(weights, dtype=np.float32).reshape((size, 1))
    
    best_particle_idx = idx[0]
    initial_centroids = FlatParticle.get_landmarks(particles, best_particle_idx)
    print("prep time:", time.time() - start)

    _, centroids = kmeans(X, weights, initial_centroids)

    return centroids




def maha_mean(xs, covs):
    # s = [x @ np.linalg.pinv(covs[i]) for i,x in enumerate(xs)]
    # c = [np.linalg.pinv(cov) for cov in covs]

    # np.sum(s, axis=0) @ np.sum()
    pass


def make_random_maps(ground_truth, sigma_min, sigma_max, n=4):
    maps = []

    sigmas = np.linspace(sigma_min, sigma_max, n)

    for i in range(n):
        m = np.copy(ground_truth)
        m += np.random.normal(0, sigmas[i], size=ground_truth.shape)
        maps.append(m)


    return maps



if __name__ == '__main__':
    ground_truth_map = np.array([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 3],
        [4, 4],
        [2, 3],
    ], dtype=np.float32)


    fig, ax = plt.subplots()
    plot_landmarks(ax, ground_truth_map)

    N = 5
    sigma_min = 0.05
    sigma_max = 0.3

    cmap = plt.cm.get_cmap("hsv", N)
    maps = make_random_maps(ground_truth_map, sigma_min=sigma_min, sigma_max=sigma_max, n=N)

    # for i, m in enumerate(maps):
    #     plot_landmarks(ax, m, color=cmap(i))


    size = N * ground_truth_map.shape[0]
    X = []
    # weights = np.ones((size, 1))
    weights = []


    w = np.linspace(1, 0, N)
    for i in range(N):
        for point in maps[i]:
            X.append(point)
            weights.append(w[i])

    X = np.array(X, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32).reshape((size, 1))

    # print(weights)

    assignment, centroids = kmeans(X, weights, maps[0])


    plot_landmarks(ax, centroids, color="orange")
    plt.show()
    


