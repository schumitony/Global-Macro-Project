import numpy as np
from joblib import Parallel, delayed
import time

class Toto:

    # if __name__ == '__main__':

    def __init__(self):
        n_vectors = 100000
        t0 = time.perf_counter()
        random_vector = [Toto.stochastic_function(1000, k) for k in range(1, n_vectors)]
        print(str(time.perf_counter() - t0))
        # print('\nThe different generated vectors in a sequential manner are:\n {}'.format(np.array(random_vector)))

        # backend = 'loky'
        t0 = time.perf_counter()
        random_vector_P = Parallel(n_jobs=-1)(delayed(Toto.stochastic_function)(1000, k) for k in range(1, n_vectors))
        print(str(time.perf_counter() - t0))
        a = 1
        # Toto.print_vector(random_vector, backend)

    @staticmethod
    def stochastic_function(n, k):
        """Randomly generate integer up to a maximum value."""
        # x = np.random.randint(10, size=n)
        li = []
        for i in range(0, n):
            li.append(i*k)

        return li

    @staticmethod
    def print_vector(vector, backend):
        """Helper function to print the generated vector with a given backend."""
        print('\nThe different generated vectors using the {} backend are:\n {}'.format(backend, np.array(vector)))


class Hop:
    A = Toto()
