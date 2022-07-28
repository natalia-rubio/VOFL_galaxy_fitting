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

        slope = (s_guess1 - s_guess2)/(K2 - K1)
        s_guess1 = copy.copy(s_guess2); K1 = copy.copy(K2)
        s_guess2 += slope*(K_target - K2); K2 = get_K(r, s_guess2, n)
        its +=1
        if its > 1000:
            print(f"Reached 1000 iterations, stopping secant method.\nAchieved K = {K2}, but target K was {K_target}.")
            break

    return s_guess2

@tf.function
def get_grad(r, v, n, x):
    with tf.GradientTape() as tape_hess:
        with tf.GradientTape() as tape_jac:
            with tf.GradientTape(persistent = True) as tape_r:
                tape_r.watch(r); tape_jac.watch(x); tape_hess.watch(x)

                s = tf.math.add(
                    tf.math.divide(tf.math.add(x[0], tf.math.multiply(x[1], r)),
                    tf.math.add(tf.convert_to_tensor(1, dtype = "float32"), tf.math.multiply(x[2], r))),
                    x[3])

                gamma_num = tf.math.lgamma(tf.math.subtract(tf.math.divide(n,tf.convert_to_tensor(2, dtype = "float32")), s))
                pi_pow = tf.math.pow(tf.convert_to_tensor(np.pi, dtype = "float32"), tf.math.divide(n, tf.convert_to_tensor(2, dtype = "float32")))
                four_pow = tf.math.pow(tf.convert_to_tensor(4, dtype = "float32"), s)
                gamma_den = tf.math.lgamma(s)
                abs_den = tf.math.pow(r, tf.math.subtract(n, tf.multiply(tf.convert_to_tensor(2, dtype = "float32"), s)))
                den = tf.multiply(tf.multiply(pi_pow, four_pow), tf.multiply(gamma_den, abs_den))

                K = tf.divide(gamma_num, den)

                # import pdb; pdb.set_trace()
            dKdr = tape_r.gradient(K, r)
            #import pdb; pdb.set_trace()
            residual_norm = tf.math.reduce_euclidean_norm(tf.subtract(tf.multiply(dKdr, r), tf.math.square(v)))

        jacobian = tape_jac.jacobian(residual_norm, x)
    hessian = tape_hess.jacobian(jacobian, x)
    return residual_norm, jacobian, hessian

def get_s(galaxy_dict):
    r = tf.constant(galaxy_dict["r"].astype(np.float32))
    r = r / tf.math.reduce_max(r)
    v = tf.constant(galaxy_dict["v"].astype(np.float32))
    v = v / tf.math.reduce_max(v)
    n = tf.constant(np.asarray([3.0,]).astype(np.float32))
    x = np.asarray([0.6, 0.9, 1, 0])
    x = tf.convert_to_tensor(x, dtype = "float32")
    residual_norm = 1
    tol = 0.1
    nits = 0
    get_grad_concrete = get_grad.get_concrete_function(r, v, n, x)
    while residual_norm > tol:
        nits += 1
        residual_norm, jacobian, hessian = get_grad_concrete(r, v, n, x)

        x = x - tf.reshape(tf.linalg.solve(hessian, tf.reshape(jacobian, [4,1])),[4,])

        residual_norm = residual_norm.numpy()
        print(f"Newton iteration {nits}.  Residual norm: {residual_norm}. X = {x.numpy()}")
    return x

if __name__ == "__main__":
    NGC_galaxies_results_dict = load_dict("NGC_galaxies_dict")
    for galaxy_name in NGC_galaxies_results_dict.keys():
        galaxy_dict = NGC_galaxies_results_dict[galaxy_name]
        s = get_s(galaxy_dict)
        pdb.set_trace()
