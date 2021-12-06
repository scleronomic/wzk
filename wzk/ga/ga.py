import numpy as np

class GA:

    def __init__(self, pop_size, n_var,
                 n_gen=10,
                 n_keep_best=None,
                 tourney_size=None,
                 mut_prob=None, mut_frac=None):

        self.n_var = n_var
        self.n_gen = n_gen
        self.tourney_size = int(max(np.ceil(pop_size / 10), 2)) if tourney_size is None else tourney_size
        self.n_keep_best = int(np.floor(pop_size / 10)) if n_keep_best is None else n_keep_best
        self.mut_prob = mut_prob if mut_frac is None else 1 - (1 - mut_frac) ^ (1 / self.n_var)

        self.hall_of_fame = np.empty((n_gen + 1, k), dtype=int)
        self.hall_of_all = np.empty((n_gen + 1, pop_size, k), dtype=int)
        self.fitness_best = np.empty((n_gen + 1))
        self.fitness_avg = np.empty((n_gen + 1))
