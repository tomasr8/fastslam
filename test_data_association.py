import unittest
import numpy as np
from data_association import assign, remove_unlikely_associations, find_unassigned_measurement_idx


class TestDataAssociation(unittest.TestCase):

    def test_assign_1(self):
        dist = np.array([
            [10, 5],
            [5, 1]
        ], dtype=np.float)

        assignment_lm, assignment_ml, cost = assign(dist)

        self.assertDictEqual(assignment_lm, {
            0: 0,
            1: 1
        })

        self.assertDictEqual(assignment_ml, {
            0: 0,
            1: 1
        })

        self.assertEqual(cost, 11)

    def test_assign_2(self):
        dist = np.array([
            [1]
        ], dtype=np.float)

        assignment_lm, assignment_ml, cost = assign(dist)

        self.assertDictEqual(assignment_lm, {
            0: 0
        })

        self.assertDictEqual(assignment_ml, {
            0: 0
        })

        self.assertEqual(cost, 1)

    def test_remove_unlikely_1(self):
        assignment = {
            0: 0,
            1: 2,
            2: 1
        }

        dist = np.array([
            [10, 10, 10],
            [10, 10, 10],
            [10, 10, 10]
        ], dtype=np.float)

        threshold = 20

        new_assignment = remove_unlikely_associations(
            assignment, dist, threshold)

        self.assertDictEqual(new_assignment, {})

    def test_remove_unlikely_2(self):
        assignment = {
            0: 0,
            2: 1
        }

        dist = np.array([
            [10, 10],
            [10, 10],
            [10, 0]
        ], dtype=np.float)

        threshold = 5

        new_assignment = remove_unlikely_associations(
            assignment, dist, threshold)

        self.assertDictEqual(new_assignment, {
            0: 0
        })

    def test_find_unassigned_1(self):
        assignment = {
            0: 0,
            1: 3
        }

        idx = find_unassigned_measurement_idx(assignment, 4)
        self.assertSetEqual(idx, set([1, 2]))


    def test_find_unassigned_1(self):
        assignment = {
            0: 0,
            1: 1
        }

        idx = find_unassigned_measurement_idx(assignment, 2)
        self.assertSetEqual(idx, set([]))

if __name__ == '__main__':
    unittest.main()
