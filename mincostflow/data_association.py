import math
import numpy as np


def assign(dist):
    M, N = dist.shape

    tuples = [(i, j, dist[i, j]) for i in range(M) for j in range(N)]
    tuples.sort(key=lambda t: t[2])
    tuples = [(t[0], t[1]) for t in tuples]

    assigned_a = set()
    assigned_b = set()
    assigned_total = 0
    assignment = {}

    cost = 0

    for i in range(M*N):
        a, b = tuples[i]

        if a in assigned_a or b in assigned_b:
            continue
        else:
            assignment[a] = b
            assigned_total += 1
            assigned_a.add(a)
            assigned_b.add(b)
            cost += dist[a, b]

        if assigned_total == M:
            break

    return assignment, cost




if __name__ == "__main__":
    M = 30
    N = 30
    np.random.seed(0)
    dist = np.random.uniform(0, 1, size=(M, N))
    np.save("in.npy", dist)
    # dist = np.array([
    #     [1, 2, 10],
    #     [1, 10, 10],
    #     [10, 10, 1]   
    # ])

    assignment, cost = assign(dist)
    print(assignment)
    print("Cost:", cost)