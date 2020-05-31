import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


print(scipy.stats.multivariate_normal.pdf([0.1, 0], mean=[0,0], cov=[[1, 0], [0, 1]]))
print(scipy.stats.multivariate_normal.pdf([0, -1], mean=[0,0], cov=[[1, 0], [0, 1]]))


print(scipy.stats.multivariate_normal.pdf([0, 2], mean=[0,0], cov=[[2, 0], [0, 2]]))
