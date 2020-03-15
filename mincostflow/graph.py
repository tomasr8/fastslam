import numpy as np

class Matrix(object):
    def __init__(self, N, flow, cost):
        self.N = N
        self.matrix = np.zeros((N, N, 2), dtype=np.float)

        self.matrix[:, :, 0] = flow
        self.matrix[:, :, 1] = cost


    def add_flow(self, u, v, f):
        self.matrix[u, v, 0] += f


    def collect_edges(self):
        edges = []

        for u in self.graph:
            for v in self.graph[u]:
                flow, cost = self.graph[u][v]
                edges.append([u, v, flow, cost])

        return edges


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
    def __init__(self):
        pass

    @staticmethod
    def collect_edges(matrix: Matrix):
        edges = []
        N = matrix.N

        for i in range(N):
            for j in range(N):
                flow, cost = matrix.matrix[i, j]

                uf = 1-flow
                if uf != 0:
                    edges.append((i, j + N, uf, cost))

                uf = flow-0
                if uf != 0:
                    edges.append((j + N, i, uf, -cost))
                    
        for i in range(N):
            edges.append((2*N, i, 0, 0))
            edges.append((2*N, i + N, 0, 0))

        return edges


# class DiGraph(object):
#     def __init__(self, V, E):
#         self.V = V
#         self.E = E
#         self.graph = {}

#     def add_edge(self, u, v, lower, upper, flow, cost):
#         if u not in self.graph:
#             self.graph[u] = {}

#         self.graph[u][v] = [lower, upper, flow, cost]


#     def add_flow(self, u, v, f):
#         self.graph[u][v][2] += f


#     def collect_edges(self):
#         edges = []

#         for u in self.graph:
#             for v in self.graph[u]:
#                 lower, upper, flow, cost = self.graph[u][v]
#                 edges.append([u, v, lower, upper, flow, cost])

#         return edges


# class ResidualGraph(object):
#     def __init__(self, graph: DiGraph):
#         self.V = graph.V
#         self.E = 2 * graph.E
#         self.graph = {}

#         for u in graph.graph:
#             if u not in self.graph:
#                 self.graph[u] = {}

#             for v in graph.graph[u]:
#                 if v not in self.graph:
#                     self.graph[v] = {}

#                 lower, upper, flow, cost = graph.graph[u][v]
#                 self.graph[u][v] = [upper-flow, cost, flow]
#                 self.graph[v][u] = [flow-lower, -cost, flow]


#     def add_dummy_source(self):
#         source = {}
#         for u in self.graph:
#             source[u] = [0, 0, 0]
#             self.E += 1

#         self.graph[self.V] = source
#         self.V += 1


#     def remove_dummy_source(self):
#         source = self.graph.pop(self.V - 1)
#         self.V -= 1
#         self.E -= len(source.keys())


#     def remove_zero_uf(self):
#         to_remove = []
#         for u in self.graph:
#             for v in self.graph[u]:
#                 uf, _, _ = self.graph[u][v]
#                 if uf == 0:
#                     to_remove.append((u, v))

#         for u, v in to_remove:
#             self.graph[u].pop(v)


#     def add_flow(self, u, v, f):
#         if v in self.graph[u]:
#             self.graph[u][v][2] += f


#     def collect_edges(self):
#         edges = []

#         for u in self.graph:
#             for v in self.graph[u]:
#                 uf, cf, flow = self.graph[u][v]
#                 edges.append([u, v, uf, cf, flow])

#         return edges