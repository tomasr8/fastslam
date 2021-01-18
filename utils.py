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