import numpy as np
from scipy.special import gamma
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy
from os.path import exists
from sklearn.cluster import KMeans
font = {"size": 16}

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_K(r, s, n):
    return gamma(n/2 - s)*((4**s) * (np.pi**(n/2)) * (r**(n-2*s)))**-1

def secant_method(K_target, r, n):
    tol = abs(K_target)/1000; its = 0
    s_guess1 = 0.5; K1 = get_K(r, s_guess1, n)
    s_guess2 = 0.8; K2 = get_K(r, s_guess2, n)
    #pdb.set_trace()
    while np.linalg.norm(K_target - K2) > tol:
        pdb.set_trace()
        slope = (s_guess1 - s_guess2)/(K2 - K1)
        s_guess1 = copy.copy(s_guess2); K1 = copy.copy(K2)
        s_guess2 += slope*(K_target - K2); K2 = get_K(r, s_guess2, n)
        its +=1
        if its > 1000:
            print(f"Reached 1000 iterations, stopping secant method.\nAchieved K = {K2}, but target K was {K_target}.")
            break

    return s_guess2

def get_s(galaxy_dict):
    r = galaxy_dict["r"]
    v = galaxy_dict["v"]
    tape
    K = -np.divide(np.power(v, 2), r)
    s_list = []
    for i in range(K.size):
        s_list.append(secant_method(K[i], r[i], 3))

    return np.asarray(s_list)
