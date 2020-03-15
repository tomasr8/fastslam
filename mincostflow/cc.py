from bf import find_neg_cycle
from graph import Matrix, ResidualMatrix


def cycle_canceling(matrix: Matrix):

    iters = 0
    while True:
        residual_edges = ResidualMatrix.to_edges(matrix)
        source = matrix.M + matrix.N

        delta, cycle = find_neg_cycle(residual_edges, matrix.M, matrix.N)

        if delta > 0:
            length = len(cycle)
            for i in range(length - 1):
                u = cycle[i]
                v = cycle[i + 1]

                if u < matrix.M:  # forward edge
                    matrix.add_flow(u, v - matrix.M, delta)
                else:
                    matrix.add_flow(v, u - matrix.M, -delta)
        else:
            break

        iters += 1

    print("Iterations:", iters)

    return matrix.recover_assignment()
