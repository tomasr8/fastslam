import math
import numpy as np
import scipy


def assign(dist):
    M, N = dist.shape

    # print(dist)
    tuples = [(i, j, dist[i, j]) for i in range(M) for j in range(N)]
    # maximize
    tuples.sort(key=lambda t: -t[2])
    # print(tuples)
    tuples = [(t[0], t[1]) for t in tuples]

    assigned_a = set()
    assigned_b = set()
    assigned_total = 0
    assignment_lm = {}
    assignment_ml = {}

    cost = 0

    for i in range(M*N):
        a, b = tuples[i]

        if a in assigned_a or b in assigned_b:
            continue
        else:
            assignment_lm[a] = b
            assignment_ml[b] = a

            assigned_total += 1
            assigned_a.add(a)
            assigned_b.add(b)
            cost += dist[a, b]

        if assigned_total == M:
            break

    return assignment_lm, assignment_ml, cost


def get_dist_matrix(landmarks, measurements, landmarks_cov, measurement_cov):
    M = len(landmarks)
    N = len(measurements)
    dist = np.zeros((M, N), dtype=np.float)

    for i in range(M):
        for j in range(N):
            cov = landmarks_cov[i] + measurement_cov
            dist[i, j] = scipy.stats.multivariate_normal.pdf(
                landmarks[i], mean=measurements[j], cov=cov, allow_singular=True)

    return dist


def remove_unlikely_associations(assignment, dist, threshold):
    M, N = dist.shape
    new_assignment = {}

    for i in range(M):
        if i in assignment:
            j = assignment[i]
            if dist[i, j] > threshold:
                new_assignment[i] = j

    return new_assignment


def find_unassigned_measurement_idx(assignement, n_measurements):
    all_idx = set(np.arange(n_measurements))
    assigned_idx = set(assignement.values())
    unassigned_idx = all_idx.difference(assigned_idx)

    return unassigned_idx


def associate_landmarks_measurements(particle, measurements, measurement_cov, threshold):
    landmarks = particle.landmark_means
    landmarks_cov = particle.landmark_covariances

    M = len(landmarks)
    N = len(measurements)

    pos = np.array([particle.x, particle.y])
    measurement_predicted = landmarks - pos

    dist = get_dist_matrix(measurement_predicted, measurements,
                           landmarks_cov, measurement_cov)
    assignment_lm, assignment_ml, _ = assign(dist)

    assignment = remove_unlikely_associations(assignment_lm, dist, threshold)
    unassigned_measurement_idx = find_unassigned_measurement_idx(assignment, N)

    # particle.add_landmarks(new_landmarks, measurement_cov)
    return assignment, unassigned_measurement_idx


if __name__ == "__main__":
    M = 2
    N = 2
    np.random.seed(0)
    dist = np.random.uniform(0, 1, size=(M, N))
    # np.save("in.npy", dist)
    # dist = np.array([
    #     [1, 2, 10],
    #     [1, 10, 10],
    #     [10, 10, 1]
    # ])

    assignment_lm, assignment_ml, cost = assign(dist)
    print(assignment_lm, assignment_ml)
    print("Cost:", cost)
