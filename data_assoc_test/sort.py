import numpy as np
import time

array = np.random.uniform(0, 1, size=5000000)
start = time.time()
np.sort(array)
print(time.time() - start)
