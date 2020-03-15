from bf import find_neg_cycle
from graph import Matrix, ResidualMatrix


def cycle_canceling(matrix: Matrix):

    iters = 0
    while True:
        residual_edges = ResidualMatrix.collect_edges(matrix)
        source = 2*matrix.N

        delta, cycle = find_neg_cycle(residual_edges, source, matrix.N)

        if delta > 0:
            length = len(cycle)
            for i in range(length - 1):
                u = cycle[i]
                v = cycle[i + 1]

                if u < matrix.N: # forward edge
                    matrix.add_flow(u, v - matrix.N, delta)
                else:
                    matrix.add_flow(v, u - matrix.N, -delta)

        else:
            break

        # if iters == 3:
        #     break

        iters += 1

    print("Iterations:", iters)

    return matrix.recover_assignment()