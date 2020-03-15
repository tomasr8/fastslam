import numpy as np
import math
import time


def find_neg_cycle(edges: list, M: int, N: int):
    V = M+N + 1
    E = 2*M*N + (M+N)
    source = M+N

    dist = np.full(V, np.inf)
    dist[source] = 0
    parent = np.full((V, 2), -1)

    start = time.time()
    for _ in range(E):
        updated = False
        last_updated = -1

        for u, v, uf, cf in edges:
            if dist[v] > dist[u] + cf:
                dist[v] = dist[u] + cf

                updated = True
                last_updated = v
                parent[v, :] = [u, uf]

        if not updated:
            break

    # print("BF:", time.time() - start)
    if last_updated == -1:
        return [0, None]

    y = last_updated
    for _ in range(V):
        y = parent[y, 0]

    delta = math.inf
    cycle = []
    curr = y
    while True:
        cycle.append(curr)

        uf = parent[curr, 1]
        delta = min([delta, uf])

        if curr == y and len(cycle) > 1:
            break

        curr = parent[curr, 0]

    cycle.reverse()
    return [delta, cycle]
