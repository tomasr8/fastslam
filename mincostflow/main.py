#!/usr/bin/env python3

import sys
import time
import math
import numpy as np
from graph import Matrix
from cc import cycle_canceling
from data_association import assign

def main():
    
    np.random.seed(0)
    # dist = np.random.uniform(0, 1, size=(N, N))
    dist = np.load("in.npy")
    N = dist.shape[0]

    assignment, _ = assign(dist)
    flow = np.zeros((N, N), dtype=np.float)

    for i in range(N):
        j = assignment[i]
        flow[i, j] = 1

    # flow = np.eye(N, dtype=np.float)

    # dist = np.array([
    #     [1, 2, 10],
    #     [10, 1, 10],
    #     [10, 10, 1]
    # ], dtype=np.float)

    # flow = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ], dtype=np.float)
    
    matrix = Matrix(N, flow, dist)

    assignment, min_cost = cycle_canceling(matrix)
    print("Cost:", min_cost)
    print(assignment)


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time:", (time.time()-start))