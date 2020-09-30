import time
import numpy as np

def pinv(A):
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]

    e = a*a + c*c
    f = a*b + c*d
    g = a*b + c*d
    h = b*b + d*d

    scalar = 1/(e*h - f*g)
    e_i = scalar * h
    f_i = scalar * (-f)
    g_i = scalar * (-g)
    h_i = scalar * e

    return np.array([
        [e_i*a + f_i*b, e_i*c + f_i*d],
        [g_i*a + h_i*b, g_i*c + h_i*d]
    ], dtype=np.float)


matrices = np.random.randint(0, 5, size=(10000, 2, 2))
start = time.time()

for M in matrices:
    pinv(M)

print(time.time() - start)