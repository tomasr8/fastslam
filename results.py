import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

xs = [256, 512, 1024, 2048, 4096, 8192]
ys_exact = np.array([
    [ 0.20234051298458627, 0.09572642888103693 ],
    [ 0.1781915457261391, 0.08735934960972884 ],
    [ 0.1610538317159449, 0.06222602647756702 ],
    [ 0.14862146246637267, 0.0583388090775533 ],
    [ 0.14258670219160055, 0.054587606366649374 ],
    [ 0.134175122582712, 0.05374129711211663 ]
])

ys_jacobi = np.array([
    [ 0.44686667238570194, 0.2880181188226436 ],
    [ 0.40569256699181233, 0.2179156964158653 ],
    [ 0.3958582228846586, 0.22516262355906466 ],
    [ 0.3452954834429051, 0.20107433006703776 ],
    [ 0.25803503559786933, 0.13448731892074145 ],
    [ 0.1952510387965101, 0.10252680480576697 ]
])


ax.errorbar(xs, ys_jacobi[:, 0], yerr=ys_jacobi[:, 1], fmt="o--", label=r"$z = (r, \phi)$")
ax.errorbar(xs, ys_exact[:, 0], yerr=ys_exact[:, 1], fmt="o--", label=r"$z = (\Delta x, \Delta y)$")

ax.set_ylabel("MSE")

plt.legend(fontsize=13)
plt.show()