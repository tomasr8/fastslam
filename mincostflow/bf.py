import numpy as np
import math
import time

def find_neg_cycle(edges: list, source, N):
    V = 2*N + 1
    E = 2*N*N + 2*N
    source = 2*N


    dist = np.full(2*N + 1, np.inf)
    dist[source] = 0
    parent = np.full((2*N + 1, 2), -1)


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

    min_uf = math.inf
    cycle = []
    curr = y
    while True:
        cycle.append(curr)

        uf = parent[curr, 1]
        min_uf = min([min_uf, uf])

        if curr == y and len(cycle) > 1:
            break

        curr = parent[curr, 0]

    cycle.reverse()
    return [min_uf, cycle]