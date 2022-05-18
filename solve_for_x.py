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
import tensorflow as tf
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

def get_grad(r, v, n, x1, x2):
    with tf.GradientTape(persistent = True) as tape0, tf.GradientTape() as tape1:
        tape0.watch(x1); tape0.watch(x2); tape1.watch(r)
        s = tf.math.divide(tf.math.add(x1, tf.math.multiply(x2, r)), tf.math.add(tf.convert_to_tensor(1, dtype = "float32") , r))
        K = tf.math.divide(tf.math.lgamma(tf.math.subtract(tf.math.divide(n,tf.convert_to_tensor(2, dtype = "float32")), s)),
        tf.multiply(tf.math.lgamma(s), tf.multiply(tf.math.pow(tf.convert_to_tensor(np.pi, dtype = "float32"), tf.math.divide(n,tf.convert_to_tensor(2, dtype = "float32"))),
        tf.multiply(tf.math.pow(tf.convert_to_tensor(4, dtype = "float32"), s),
        tf.math.pow(r, tf.math.subtract(n, tf.multiply(tf.convert_to_tensor(2, dtype = "float32"),s)))))))
        dKdr = tape1.gradient(K, r)
        import pdb; pdb.set_trace()
        residual_norm = tf.math.reduce_euclidean_norm(tf.subtract(tf.multiply(dKdr, r), tf.math.square(v)))
        grad_x1 = tape0.gradient(residual_norm, x1)
        grad_x2 = tape0.gradient(residual_norm, x2)
    return residual_norm, grad_x1, grad_x2

def get_s(galaxy_dict):
    r = tf.constant(galaxy_dict["r"].astype(np.float32))
    v = tf.constant(galaxy_dict["v"].astype(np.float32))
    n = tf.constant([3], dtype = "float32")
    x1 = tf.Variable([1], dtype = "float32"); x2 = tf.Variable([1], dtype = "float32")
    residual_norm = 1
    tol = 0.1
    while residual_norm > tol:
        residual_norm, grad_x1, grad_x2 = get_grad(r, v, n, x1, x2)
        x1 = x1 - tf.divide(residual_norm, grad_x1)
        x2 = x2 - tf.divide(residual_norm, grad_x2)
        residual_norm = residual_norm.numpy()
        print(f"residual norm: {residual_norm}")
        import pdb; pdb.set_trace()
    return x1, x2
if __name__ == "__main__":
    NGC_galaxies_results_dict = load_dict("NGC_galaxies_dict")
    for galaxy_name in NGC_galaxies_results_dict.keys():
        galaxy_dict = NGC_galaxies_results_dict[galaxy_name]
        s = get_s(galaxy_dict)
        pdb.set_trace()
