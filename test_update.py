import numpy as np
# from data_association import associate_landmarks_measurements
from particle_slam_limited_unknown import update as update_ref
from particle_slam_limited_unknown_2 import update as update_cuda
from particle import Particle

def run(fn):
    p = Particle(1.0, 2.0, 0.15, 0, 0.5)
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

    particles = [p]

    measurements = np.array([
        [10.2, 20.1],
        [12.1, 21.95]
    ], dtype=np.float32)

    measurement_var = [0.05, 0.1]
    threshold = 0.1

    print(p.n_landmarks)

    # for threshold in np.linspace(0, 1, num=10):
        # print(associate_landmarks_measurements(p, measurements, measurement_cov, threshold))

    fn(particles, measurements, measurement_var)

    for p in particles:
        # print("x y θ", p.x, p.y, p.theta)
        print(f"w: {p.w}, n_landmarks: {p.n_landmarks}")
        print(p.landmark_means)



run(update_ref)
run(update_cuda)


