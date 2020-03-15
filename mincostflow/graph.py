import numpy as np


class Matrix(object):
    def __init__(self, M, N, flow, cost):
        self.M = M
        self.N = N
        self.matrix = np.zeros((M, N, 2), dtype=np.float)

        self.matrix[:, :, 0] = flow
        self.matrix[:, :, 1] = cost

    def add_flow(self, u, v, f):
        self.matrix[u, v, 0] += f

    def recover_assignment(self):
        assignment = {}
        min_cost = 0

        for i in range(self.N):
            for j in range(self.N):
                flow, cost = self.matrix[i, j]

                if flow == 1:
                    assignment[i] = j
                    min_cost += cost
                    break

        return assignment, min_cost


class ResidualMatrix(object):

    @staticmethod
    def to_edges(matrix: Matrix):
        edges = []
        M = matrix.M
        N = matrix.N

        for i in range(M):
            for j in range(N):
                flow, cost = matrix.matrix[i, j]

                uf = 1-flow
                if uf != 0:
                    edges.append((i, j + M, uf, cost))

                uf = flow-0
                if uf != 0:
                    edges.append((j + M, i, uf, -cost))

        for i in range(M):
            edges.append((M + N, i, 0, 0))
            edges.append((M + N, i + M, 0, 0))

        return edges
