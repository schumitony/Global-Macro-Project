import numpy as np
from joblib import Parallel, delayed
import time

class Toto:

    # if __name__ == '__main__':

    def __init__(self):
        n_vectors = 500000
        t0 = time.perf_counter()
        random_vector = [Toto.stochastic_function(10) for _ in range(n_vectors)]
        print(str(time.perf_counter() - t0))
        print('\nThe different generated vectors in a sequential manner are:\n {}'.format(np.array(random_vector)))

        backend = 'loky'
        t0 = time.perf_counter()
        random_vector = Parallel(n_jobs=4, backend=backend)(delayed(
            Toto.stochastic_function)(10) for _ in range(n_vectors))
        print(str(time.perf_counter() - t0))
        Toto.print_vector(random_vector, backend)

    @staticmethod
    def stochastic_function(max_value):
        """Randomly generate integer up to a maximum value."""
        return np.random.randint(max_value, size=5)

    @staticmethod
    def print_vector(vector, backend):
        """Helper function to print the generated vector with a given backend."""
        print('\nThe different generated vectors using the {} backend are:\n {}'.format(backend, np.array(vector)))


class Hop:
    A = Toto()