import math
import numpy as np


def assign(dist):
    N, M = dist.shape

    tuples = [(i, j, dist[i, j]) for i in range(N) for j in range(M)]
    tuples.sort(key=lambda t: t[2])
    tuples = [(t[0], t[1]) for t in tuples]

    assigned_a = set()
    assigned_b = set()
    assigned_n = 0
    assignment = {}

    cost = 0

    for i in range(N*M):
        a, b = tuples[i]

        if a in assigned_a or b in assigned_b:
            continue
        else:
            assignment[a] = b
            assigned_n += 1
            assigned_a.add(a)
            assigned_b.add(b)
            cost += dist[a, b]

        if assigned_n == N:
            break

    return assignment, cost




if __name__ == "__main__":
    N = 30
    M = 30
    np.random.seed(0)
    dist = np.random.uniform(0, 1, size=(N, M))
    np.save("in.npy", dist)
    # dist = np.array([
    #     [1, 2, 10],
    #     [1, 10, 10],
    #     [10, 10, 1]   
    # ])

    assignment, cost = assign(dist)
    print(assignment)
    print("Cost:", cost)