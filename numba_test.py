import numpy as np
import scipy
import numba as nb


# def get_dist_matrix(landmarks, measurements, landmarks_cov, measurement_cov):
#     M = len(landmarks)
#     N = len(measurements)
#     dist = np.zeros((M, N), dtype=np.float)

#     for i in range(M):
#         for j in range(N):
#             cov = landmarks_cov[i] + measurement_cov

#             dist[i, j] = scipy.stats.multivariate_normal.pdf(
#                 landmarks[i], mean=measurements[j], cov=cov, allow_singular=False)

#     return dist


@nb.njit
def numpy_matrix_test(x, y):
    M = len(x)
    N = len(y)
    A = np.zeros((M, N), dtype=np.float32)
    return A



if __name__ == '__main__':
    numpy_matrix_test(np.array([1, 2]), np.array([1, 2]))
    print("sdfdsfsd")