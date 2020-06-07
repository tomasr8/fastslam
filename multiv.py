import time
import numpy as np
import scipy.stats
import math


def pdf(x, mean, cov):
    return np.exp(logpdf(x, mean, cov))


def logpdf(x, mean, cov):
    # `eigh` assumes the matrix is Hermitian.
    # print("cov", cov)
    vals, vecs = np.linalg.eigh(cov)
    # print(vals, vecs)
    logdet     = np.sum(np.log(vals))
    valsinv    = np.array([1./v for v in vals])
    # `vecs` is R times D while `vals` is a R-vector where R is the matrix 
    # rank. The asterisk performs element-wise multiplication.
    # print(vecs.shape)
    # print(valsinv.shape)
    U          = vecs * np.sqrt(valsinv)
    # print(U.shape)
    # print("===")
    rank       = len(vals)
    dev        = x - mean
    # "maha" for "Mahalanobis distance".
    maha       = np.square(np.dot(dev, U)).sum()
    log2pi     = np.log(2 * np.pi)
    return -0.5 * (rank * log2pi + maha + logdet)


def pdf_2(x, mean, cov):
    a, b = cov[0, :]

    logdet = math.log(a*a - b*b)

    root = math.sqrt(2)/2
    e = root * (a-b)**(-0.5)
    f = root * (a+b)**(-0.5)

    m = x[0] - mean[0]
    n = x[1] - mean[1]

    maha = 2*(m*m*e*e + n*n*f*f)
    log2pi = math.log(2 * math.pi)
    return math.exp(-0.5 * (2*log2pi + maha + logdet))



if __name__ == '__main__':
    x = np.array([2.07251885, 1.18032621], dtype=np.float)
    mean = np.array([-1.24014031, 2.37559755], dtype=np.float)

    cov = np.array([
        [0.2, 0],
        [0, 0.2]
    ], dtype=np.float)


    # a = np.array([
    #     [2, 5],
    #     [5, 2]
    # ], dtype=np.float)

    # x = np.array([3, 1], dtype=np.float)

    # print(np.dot(x, a))


    print(scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=False))
    print(pdf(x, mean, cov))
    log2pi     = np.log(2 * np.pi)
    print(pdf_2(x, mean, cov))



    start_time = time.time()
    for _ in range(10000):
        scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=False)

    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()

    for _ in range(10000):
        pdf(x, mean, cov)

    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    log2pi     = np.log(2 * np.pi)
    for _ in range(10000):
        pdf_2(x, mean, cov)

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
