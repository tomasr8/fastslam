import numpy as np

class Sensor(object):
    def __init__(self, landmarks, phantom_landmarks, measurement_variance, range, fov, miss_prob, phantom_prob, rb=False):
        self.landmarks = landmarks
        self.phantom_landmarks = phantom_landmarks
        self.measurement_variance = measurement_variance
        self.range = range
        self.fov = fov
        self.miss_prob = miss_prob
        self.phantom_prob = phantom_prob
        self.rb = rb


    def __get_noisy_measurement(self, position, landmark):
        vector_to_landmark = np.array(landmark - position, dtype=np.float32)

        a = np.random.normal(0, self.measurement_variance[0])
        b = np.random.normal(0, self.measurement_variance[1])
        vector_to_landmark[0] += a
        vector_to_landmark[1] += b

        return vector_to_landmark

    def __get_noisy_rb_measurement(self, pose, landmark):
        position = pose[:2]
        vector_to_landmark = np.array(landmark - position, dtype=np.float32)

        r = np.linalg.norm(vector_to_landmark)
        b = np.arctan2(vector_to_landmark[1], vector_to_landmark[0]) - pose[2]

        r += np.random.normal(0, self.measurement_variance[0])
        b += np.random.normal(0, self.measurement_variance[1])

        return [r, b]


    def get_noisy_measurements(self, pose):
        measurements = {
            "observed": np.zeros((0, 2), dtype=np.float32),
            "missed": np.zeros((0, 2), dtype=np.float32),
            "outOfRange": np.zeros((0, 2), dtype=np.float32),
            # "phantomSeen": np.zeros((0, 2), dtype=np.float32),
            # "phantomNotSeen": np.zeros((0, 2), dtype=np.float32),
        }

        position = pose[:2]

        for i, landmark in enumerate(self.landmarks):
            if self.rb:
                z = self.__get_noisy_rb_measurement(pose, landmark)
            else:
                z = self.__get_noisy_measurement(position, landmark)

            coin_toss = np.random.uniform(0, 1)
            if self.__in_sensor_range(landmark, pose):
                if coin_toss > self.miss_prob:
                    measurements["observed"] = np.vstack((measurements["observed"], [z]))
                else:
                    measurements["missed"] = np.vstack((measurements["missed"], [landmark]))
            else:
                measurements["outOfRange"] = np.vstack((measurements["outOfRange"], [landmark]))


        for key in measurements:
            measurements[key] = measurements[key].astype(np.float32)

        return measurements


    def __in_sensor_range(self, landmark, pose):
        x, y, theta = pose

        va = [landmark[0] - x, landmark[1] - y]
        vb = [self.range * np.cos(theta), self.range * np.sin(theta)]

        if np.linalg.norm(va) > self.range:
            return False

        angle = np.arccos(np.dot(va, vb)/(np.linalg.norm(va)*np.linalg.norm(vb)))

        return angle <= (self.fov/2)
