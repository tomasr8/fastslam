import itertools
from multiprocessing import Pool, freeze_support

def func(a, b):
    print(a, b)

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return func(*a_b)

def main():
    pool = Pool()
    a_args = [1,2,3]
    second_arg = 1
    pool.map(func_star, [(1, 1), (2, 1), (3, 1)])

if __name__=="__main__":
    main()