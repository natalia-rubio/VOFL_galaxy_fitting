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
    # for simple python, not used
    return gamma(n/2 - s)*((4**s) * (np.pi**(n/2)) * (r**(n-2*s)))**-1

# tf function to get res_norm, d res_norm/dx, and d^2 res_norm/dx^2
@tf.function
def get_grad(r, v, n, x):
    with tf.GradientTape() as tape_hess: # this tape tracks the Hessian of the residual norm wrt x
        with tf.GradientTape() as tape_jac: # this tape tracks the Jacobian of the residual norm wrt x
            with tf.GradientTape(persistent = True) as tape_r:  # this tape tracks dK/dr
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

                K_fac = tf.constant()
                K = tf.divide(gamma_num, den) # get K - gravitational potential

            dKdr = tape_r.gradient(K, r)

            residual_norm = tf.math.reduce_euclidean_norm(tf.add(tf.multiply(dKdr, r), tf.math.square(v)))

        jacobian = tape_jac.jacobian(residual_norm, x)
    hessian = tape_hess.jacobian(jacobian, x)
    return residual_norm, jacobian, hessian

def get_s(galaxy_dict):
    r = tf.constant(galaxy_dict["r"].astype(np.float32)) # load in radius
    r = r / tf.math.reduce_max(r) # normalize radius
    v = tf.constant(galaxy_dict["v"].astype(np.float32)) # load in orbital velocity
    v = v / tf.math.reduce_max(v) # normalize orbital velocity
    n = tf.constant(np.asarray([3.0,]).astype(np.float32)) # set number of dimensions
    x = np.asarray([0.6, 0.9, 1, 0]) # initial guesses for x
    x = tf.convert_to_tensor(x, dtype = "float32")
    residual_norm = 1
    tol = 0.1 # acceptable residual norm
    nits = 0 # number of Newton iterations
    get_grad_concrete = get_grad.get_concrete_function(r, v, n, x) # initialize tensorflow concrete function
    while residual_norm > tol:
        nits += 1

        residual_norm, jacobian, hessian = get_grad_concrete(r, v, n, x) # tf function to get res_norm, d res_norm/dx, and d^2 res_norm/dx^2
        x = x - tf.reshape(tf.linalg.solve(hessian, tf.reshape(jacobian, [4,1])),[4,]) # take Newton step
        residual_norm = residual_norm.numpy()

        print(f"Newton iteration {nits}.  Residual norm: {residual_norm}. X = {x.numpy()}")
        if nits > 30:   break

    return x

if __name__ == "__main__":

    NGC_galaxies_results_dict = load_dict("NGC_galaxies_dict")
    for galaxy_name in NGC_galaxies_results_dict.keys():
        galaxy_dict = NGC_galaxies_results_dict[galaxy_name]
        s = get_s(galaxy_dict)
        pdb.set_trace()
