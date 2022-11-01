import numpy as np
from scipy.special import gamma
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy
from os.path import exists
import tensorflow as tf
from util import *
from newton_util import *
from grid_search_util import *

plt.rcParams.update({'font.size': 16})

def get_s(galaxy_dict):

    # grid_ranges = {"x0": [0.8, 1.3],
    #                 "x1": [1, 2],
    #                 "x2": [0.2, 0.8]}

    # Works well with absolute value on dPhidr
    grid_ranges = {"x0": [2.8, 4],
                    "x1": [1.5, 2],
                    "x2": [1, 1.5]}

    # grid_ranges = {"x0": [0.5, 2],
    #                 "x1": [0.5, 2],
    #                 "x2": [0.2, 1.5]}

    grid_x = grid_search(grid_ranges, galaxy_dict)
    newton_x = run_newton_solve(initial_guess = grid_x, galaxy_dict = galaxy_dict)
    import pdb; pdb.set_trace()
    # initial residual_norm, acceptable residual norm, number of Newton iterations


if __name__ == "__main__":

    NGC_galaxies_results_dict = load_dict("NGC_galaxies_dict")
    for galaxy_name in NGC_galaxies_results_dict.keys():
        galaxy_dict = NGC_galaxies_results_dict[galaxy_name]
        s = get_s(galaxy_dict)
        pdb.set_trace()
