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
    M, N = dist.shape

    assignment, _ = assign(dist)
    flow = np.zeros((M, N), dtype=np.float)

    for i in range(M):
        j = assignment[i]
        flow[i, j] = 1

    matrix = Matrix(M, N, flow, dist)

    assignment, min_cost = cycle_canceling(matrix)
    print("Cost:", min_cost)
    print(assignment)


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time:", (time.time()-start))