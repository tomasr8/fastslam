import time
import numpy as np
import scipy.stats
import math


def pinv_2(A):
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



if __name__ == '__main__':
    cov = np.array([
        [0.2, 1],
        [0, 0.2]
    ], dtype=np.float)

    print(np.linalg.pinv(cov))
    print(pinv_2(cov))




    start_time = time.time()

    for _ in range(10000):
        np.linalg.pinv(cov)

    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    for _ in range(10000):
        pinv_2(cov)

    print("--- %s seconds ---" % (time.time() - start_time))










# def logpdf_2(x, mean, cov, log2pi):
# a, b = cov[0, :]

# logdet = math.log(a*a - b*b)

# root = math.sqrt(2)/2
# e = root * (a-b)**(-0.5)
# f = root * (a+b)**(-0.5)

# m = x[0] - mean[0]
# n = x[1] - mean[1]

# g = m*e + n*(-f)
# h = m*e + n*f

# g = g*g
# h = h*h

# maha = g + h
# return -0.5 * (2*log2pi + maha + logdet)
