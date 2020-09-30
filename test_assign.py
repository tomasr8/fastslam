import numpy as np
from data_association import associate_landmarks_measurements
from particle import Particle


p = Particle(1.0, 2.0, 0.15, 3, 0.5)
p.add_landmarks(
    [
        [10.0, 20.0],
        [11.0, 21.0],
        [12.0, 22.0]
    ],
    [
        [0.1, 0.0],
        [0.0, 0.2]
    ]
)

measurements = np.array([
    [10.2, 20.1],
    [12.1, 21.95]
], dtype=np.float32)

measurement_cov = np.diag([0.05, 0.1])
# threshold = 0.1

for threshold in np.linspace(0, 1, num=10):
    print(associate_landmarks_measurements(p, measurements, measurement_cov, threshold))