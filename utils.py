import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

def dotify(dictionary):
    if not isinstance(dictionary, dict):
        return dictionary

    return dotdict({
        key: (dotify(value) if isinstance(value, dict) else value) for key, value in dictionary.items()
    })


def repeat(slam, times=100, seed=0):
    np.random.seed(seed)

    deviations = []
    for _ in range(times):
        deviation = slam(plot=False, seed=np.random.randint(0, 1e6))
        deviations.append(deviation)

    return np.mean(deviations), np.std(deviations)

