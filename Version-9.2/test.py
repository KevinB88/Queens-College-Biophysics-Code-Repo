import random as rng
import math as mat
import numpy as np
import time as clk
from scipy import stats

if __name__ == "__main__":

    u_bound = 100
    key_container = []

    # n = rng.randint(0, u_bound)

    for n in range(u_bound):

        key = 1
        print(f'Current: {n}')

        if n % 2 == 0:
            print('A')
            key *= 2
        else:
            print('B')
            key *= 3

        n_prime = mat.sqrt(n)

        if mat.ceil(n_prime) % 2 == 0:
            print('C')
            key *= 5
        else:
            print('D')
            key *= 7

        n_d_prime = (abs(mat.sin(n_prime))*u_bound)

        if mat.ceil(n_d_prime) % 2 == 0:
            print('E')
            key *= 11
        else:
            print('F')
            key *= 13

        key_container.append(key)

        print(f'Corresponding key: {key}')

    mode = stats.mode(key_container)
    print(mode)

