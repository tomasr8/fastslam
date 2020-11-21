import numpy as np
import matplotlib.pyplot as plt

in_kernel_max_theoretical_memory = 25 * 2014

particle_memory = {
    256: 1056,
    512: 2112,
    1024: 4224,
    2048: 8448,
    4096: 16896,
    8192: 33792,
    16384: 67584,
    32768: 135168
}


xs = [int(key) for key in particle_memory.keys()]
ys = [(value + in_kernel_max_theoretical_memory)/1024 for value in particle_memory.values()]

fig, ax = plt.subplots()

ax.plot(np.log2(xs), ys, linestyle="--", marker="o")
plt.xticks(ticks=np.log2(xs), labels=xs)
ax.set_xlabel("# Particles")
ax.set_ylabel("Maximum theoretical memory (MB)")

plt.show()