import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

# Sigma = np.diag([0.01, 0.01])
# R = np.diag([0.1, 0.1])
# P = np.diag([0.5, 0.5, 0.1])
# Gs = np.array([
#     [-1.0, 0, 0],
#     [0, -1.0, 0]
# ])


# est = np.array([1.0, 0, 0]).T
# zz = np.array([8.0 - 9.0, 0]).T


# Q = R + Sigma

# Sigma_proposal = np.linalg.inv((Gs.T @ np.linalg.inv(Q) @ Gs) + np.linalg.inv(P))
# mu_proposal = Sigma_proposal @ Gs.T @ np.linalg.inv(Q) @ (zz) + est

# print(mu_proposal)


Sigma = np.diag([0.01, 0.01, 0.01, 0.01])

R = np.diag([0.1, 0.1, 0.1, 0.1])
P = np.diag([0.5, 0.5, 0.1])
Gs = np.array([
    [-1.0, 0, 0],
    [0, -1.0, 0],
    [-1.0, 0, 0],
    [0, -1.0, 0]
])

est = np.array([1.0, 0, 0]).T
zz = np.array([8.0 - 9.0, -1, 8.0 - 9.0, -1]).T

Q = R + Sigma

Sigma_proposal = np.linalg.inv((Gs.T @ np.linalg.inv(Q) @ Gs) + np.linalg.inv(P))

mu_proposal = Sigma_proposal @ Gs.T @ np.linalg.inv(Q) @ (zz) + est

print(mu_proposal)
print(Sigma_proposal)