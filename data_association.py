import math
import numpy as np
import scipy
import numba as nb
from numba import cuda
import time

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


@nb.njit()
def get_dist_matrix(landmarks, measurements, landmarks_cov, measurement_cov):
    M = len(landmarks)
    N = len(measurements)
    dist = np.zeros((M, N), dtype=np.float32)

    # print("====")
    # print("m_cov", measurement_cov)
    for i in range(M):
        # print("lm_cov:", landmarks_cov[i])
        for j in range(N):
            cov = landmarks_cov[i] + measurement_cov

            # cov_inv = np.linalg.pinv(cov)
            # dist[i, j] = scipy.spatial.distance.mahalanobis(landmarks[i], measurements[j], cov_inv)

            # dist[i, j] = maha(landmarks[i], measurements[j], cov)

            # dist[i, j] = pdf(landmarks[i], mean=measurements[j], cov=cov)

            dist[i, j] = pdf_2(landmarks[i], mean=measurements[j], cov=cov)
            # print(dist[i, j], landmarks[i], measurements[j], cov)

            # dist[i, j] = scipy.stats.multivariate_normal.pdf(
                # landmarks[i], mean=measurements[j], cov=cov, allow_singular=False)

    # print("====")
    # print(dist)
    # print("====")

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

    M = landmarks.shape[0]
    N = measurements.shape[0]

    pos = np.array([particle.x, particle.y])
    measurement_predicted = landmarks - pos

    dist = get_dist_matrix(measurement_predicted, measurements,
                           landmarks_cov, measurement_cov)
    assignment_lm, assignment_ml, _ = assign(dist)

    assignment = remove_unlikely_associations(assignment_lm, dist, threshold)
    unassigned_measurement_idx = find_unassigned_measurement_idx(assignment, N)

    # particle.add_landmarks(new_landmarks, measurement_cov)
    return assignment, unassigned_measurement_idx


@nb.njit()
def pdf_2(x, mean, cov):
    a, b = cov[0, :]

    logdet = math.log(a*a - b*b)

    root = math.sqrt(2)/2
    e = root * (a-b)**(-0.5)
    f = root * (a+b)**(-0.5)

    m = x[0] - mean[0]
    n = x[1] - mean[1]

    maha = 2*(m*m*e*e + n*n*f*f)
    log2pi = math.log(2 * math.pi)
    return math.exp(-0.5 * (2*log2pi + maha + logdet))



def pdf(x, mean, cov):
    return np.exp(logpdf(x, mean, cov))


def logpdf(x, mean, cov):
    # `eigh` assumes the matrix is Hermitian.
    # print("cov", cov)
    # print(x.shape)
    # print(mean.shape)
    # print(cov.shape)

    # print(x)
    # print(mean)
    # print(cov)


    vals, vecs = np.linalg.eigh(cov)
    # print(vals, vecs)
    logdet     = np.sum(np.log(vals))
    valsinv    = np.array([1./v for v in vals])
    # `vecs` is R times D while `vals` is a R-vector where R is the matrix 
    # rank. The asterisk performs element-wise multiplication.
    U          = vecs * np.sqrt(valsinv)
    rank       = len(vals)
    dev        = x - mean
    # "maha" for "Mahalanobis distance".
    maha       = np.square(np.dot(dev, U)).sum()
    log2pi     = np.log(2 * np.pi)
    return -0.5 * (rank * log2pi + maha + logdet)


def maha(x, y, cov):
    cov_inv = np.linalg.pinv(cov)
    dif = x - y
    return np.square(np.dot(dif, cov_inv)).sum()


if __name__ == "__main__":
    start = time.time()
    M = 1000
    N = 100
    np.random.seed(0)
    dist = np.random.uniform(0, 1, size=(M, N))

    # dist = np.float32([
    #     [1, 2, 10, 3, 6],
    #     [22, 4, 6, 5, 6],
    #     [3, 2, 1, 99, 2],
    #     [2, 7, 8, 1, 6],
    #     [33, 44, 17, 11, 31]
    # ])

    # np.save("in.npy", dist)
    # dist = np.array([
    #     [1, 2, 10],
    #     [1, 10, 10],
    #     [10, 10, 1]
    # ])

    # start = time.time()
    assignment_lm, assignment_ml, cost = assign(dist)
    # print(assignment_lm, assignment_ml)
    print("Cost:", cost)
    print("Time: ", (time.time() - start))
