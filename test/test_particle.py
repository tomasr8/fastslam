import unittest
import math
import numpy as np
from particle import Particle


class TestDataAssociation(unittest.TestCase):

    def test_init(self):
        p = Particle(1, 2, math.pi, 1, 0.5)

        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)
        self.assertEqual(p.theta, math.pi)
        self.assertEqual(p.n_landmarks, 1)
        self.assertEqual(p.w, 0.5)

        self.assertTupleEqual(p.landmark_means.shape, (1, 2))
        self.assertTupleEqual(p.landmark_covariances.shape, (1, 2, 2))

        self.assertListEqual(p.landmark_means.tolist(), [
            [0.0, 0.0]
        ])

        self.assertListEqual(p.landmark_covariances.tolist(), [
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        ])


    def test_copy(self):
        p = Particle(1, 2, math.pi, 1, 0.5)
        p = p.copy()

        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)
        self.assertEqual(p.theta, math.pi)
        self.assertEqual(p.n_landmarks, 1)
        self.assertEqual(p.w, 0.5)

        self.assertTupleEqual(p.landmark_means.shape, (1, 2))
        self.assertTupleEqual(p.landmark_covariances.shape, (1, 2, 2))

        self.assertListEqual(p.landmark_means.tolist(), [
            [0.0, 0.0]
        ])

        self.assertListEqual(p.landmark_covariances.tolist(), [
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        ])


    def test_add_landmarks(self):
        p = Particle(1, 2, math.pi, 1, 0.5)
        p.add_landmarks(
            np.array([
                [1, 2],
                [5, 11]
            ]),
            np.array([
                [1, 0],
                [0, 2]
            ])
        )

        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)
        self.assertEqual(p.theta, math.pi)
        self.assertEqual(p.n_landmarks, 3)
        self.assertEqual(p.w, 0.5)

        self.assertTupleEqual(p.landmark_means.shape, (3, 2))
        self.assertTupleEqual(p.landmark_covariances.shape, (3, 2, 2))

        self.assertListEqual(p.landmark_means.tolist(), [
            [0.0, 0.0],
            [1.0, 2.0],
            [5.0, 11.0]
        ])

        self.assertListEqual(p.landmark_covariances.tolist(), [
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ],
            [
                [1.0, 0.0],
                [0.0, 2.0]
            ],
            [
                [1.0, 0.0],
                [0.0, 2.0]
            ],
        ])


if __name__ == '__main__':
    unittest.main()
