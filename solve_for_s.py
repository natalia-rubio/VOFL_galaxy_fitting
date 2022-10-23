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
font = {"size": 16}

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def load_galaxy_data(galaxy_dict):
    r = tf.constant(galaxy_dict["r"].astype(np.float64)) # load in radius
    #r = r / tf.math.reduce_max(r) # normalize radius
    v = tf.constant(galaxy_dict["v"].astype(np.float64)) # load in orbital velocity
    #v = v / tf.math.reduce_max(v) # normalize orbital velocity
    n = tf.constant(np.asarray([3.0,]).astype(np.float64)) # set number of dimensions
    x = np.asarray([0.1 , 1/np.max(galaxy_dict["r"]), 1/np.max(galaxy_dict["r"]), 0]) # initial guesses for x
    x = tf.convert_to_tensor(x, dtype = "float64")
    return r, v, n, x

def check_fin_diff(eps, r, v, n, x_base, residual_norm, jacobian, hessian, get_grad_concrete):
    """
    Use central difference to check that Jacobian and Hessian from TensorFlow are correct.
    """

    x = copy.deepcopy(x_base)
    jac_fin_dif = np.zeros(x.shape)
    hess_fin_dif = np.zeros((x.size, x.size))

    for i in range(x.size):

        x[i] = x_base[i] - eps
        residual_norm_neg, jac_neg , _ = get_grad_concrete(r, v, n, tf.convert_to_tensor(x, dtype = "float64"))
        x[i] = x_base[i] + eps
        residual_norm_pos, jac_pos , _ = get_grad_concrete(r, v, n, tf.convert_to_tensor(x, dtype = "float64"))
        x[i] = x_base[i]
        jac_fin_dif[i] = (residual_norm_pos - residual_norm_neg)/(2*eps) # ith entry of jacobian

        for j in range(x.size):
            x[j] = x_base[j] - eps
            residual_norm_neg, jac_neg , _ = get_grad_concrete(r, v, n, tf.convert_to_tensor(x, dtype = "float64"))
            x[j] = x_base[j] + eps
            residual_norm_pos, jac_pos , _ = get_grad_concrete(r, v, n, tf.convert_to_tensor(x, dtype = "float64"))
            x[j] = x_base[j]
            hess_fin_dif[i,j] = (jac_pos[i] - jac_neg[i])/(2*eps) # ith entry of jacobian

    print(f"Jacobian (TensorFlow): {jacobian}")
    print(f"Jacobian (Central Differnce): {jac_fin_dif}")
    print(f"Hessian (TensorFlow): {hessian}")
    print(f"Hessian (Central Differnce): {hess_fin_dif}")
    return

def get_K(r, s, n):
    # for simple python, not used
    return gamma(n/2 - s)*((4**s) * (np.pi**(n/2)) * (r**(n-2*s)))**-1

# tf function to get res_norm, d res_norm/dx, and d^2 res_norm/dx^2
@tf.function
def get_grad(r, v, n, x):
    tf.print("x", x)
    with tf.GradientTape() as tape_hess: # this tape tracks the Hessian of the residual norm wrt x
        with tf.GradientTape() as tape_jac: # this tape tracks the Jacobian of the residual norm wrt x
            with tf.GradientTape(persistent = True) as tape_r:  # this tape tracks dK/dr
                tape_r.watch(r); tape_jac.watch(x); tape_hess.watch(x)

                s = tf.math.add(tf.math.divide(tf.math.add(x[0], tf.math.multiply(x[1], r)),
                    tf.math.add(tf.convert_to_tensor(1, dtype = "float64"), tf.math.multiply(x[2], r))), x[3])

                gamma_num = tf.math.exp(tf.math.lgamma(tf.math.subtract(tf.math.divide(n,tf.convert_to_tensor(2, dtype = "float64")), s)))
                pi_pow = tf.math.pow(tf.convert_to_tensor(np.pi, dtype = "float64"), tf.math.divide(n, tf.convert_to_tensor(2, dtype = "float64")))
                four_pow = tf.math.pow(tf.convert_to_tensor(4, dtype = "float64"), tf.subtract(s, tf.convert_to_tensor(1, dtype = "float64")))


                gamma_den = tf.math.exp(tf.math.lgamma(s))
                abs_den = tf.math.pow(r, tf.math.subtract(n, tf.multiply(tf.convert_to_tensor(2, dtype = "float64"), s)))
                den = tf.multiply(tf.multiply(pi_pow, four_pow), tf.multiply(gamma_den, abs_den))
                length_scale = tf.math.pow(tf.math.pow(tf.convert_to_tensor(10, dtype = "float64"),\
                tf.convert_to_tensor(19 , dtype = "float64")), \
                tf.subtract(tf.convert_to_tensor(2, dtype = "float64"),\
                tf.multiply(tf.convert_to_tensor(2, dtype = "float64"), s)))
                #num = tf.multiply(gamma_num, length_scale)
                num = gamma_num
                tf.print("gamma_num: ", gamma_num)
                tf.print("gamma_num inside: ", tf.math.subtract(tf.math.divide(n,tf.convert_to_tensor(2, dtype = "float64")), s))
                tf.print("gamma_den: ", gamma_den)

                #K_fac = tf.constant(6.67*10**(-11) * 10**(300), dtype = tf.float64)
                K_fac = tf.constant(6.67*10**(-11) * 10**(39), dtype = tf.float64)
                K = tf.divide(num, den) # get K - gravitational potential

            # dKdr = tf.math.log(tape_r.gradient(K, r))
            dKdr = tape_r.gradient(K, r)

            if True:
                tf.print("K: ", K)
                tf.print("r shape: ", tf.shape(r))
                tf.print("dKdr: ", dKdr)
                tf.print("s: ", s)
                tf.print("length scale: ", length_scale)
                tf.print("r: ", r)
                tf.print("r * dKdr: ", tf.multiply(r, dKdr))
                tf.print("dKdr shape: ", tf.shape(dKdr))
                tf.print("dKdr maximum: ", tf.reduce_max(dKdr))

                tf.print("dKdr * K_fac ", tf.multiply(K_fac, tf.multiply(dKdr, r)))
                tf.print("sqrt dKdr * K_fac ", tf.sqrt(tf.multiply(K_fac, tf.multiply(dKdr, r))))
                tf.print("before norm ", tf.subtract( \
                tf.sqrt(tf.multiply(K_fac, tf.multiply(dKdr, r))), tf.math.abs(v)))

            residual_norm = tf.math.reduce_euclidean_norm(tf.subtract( \
            tf.sqrt(tf.multiply(K_fac, tf.multiply(dKdr, r))), tf.math.abs(v)))
            tf.print("residual_norm: ", residual_norm)

        jacobian = tape_jac.jacobian(residual_norm, x)
        tf.print("Jacobian: ", jacobian)
    hessian = tape_hess.jacobian(jacobian, x)
    tf.print("Hessian: ", hessian)
    return residual_norm, jacobian, hessian

def get_s(galaxy_dict):

    r, v, n, x = load_galaxy_data(galaxy_dict)
    residual_norm = 1
    tol = 0.1 # acceptable residual norm
    nits = 0 # number of Newton iterations
    get_grad_concrete = get_grad.get_concrete_function(r, v, n, x) # initialize tensorflow concrete function
    while residual_norm > tol:
        nits += 1

        residual_norm, jacobian, hessian = get_grad_concrete(r, v, n, x) # tf function to get res_norm, d res_norm/dx, and d^2 res_norm/dx^2
        #check_fin_diff(0.001, r, v, n, x.numpy(), residual_norm, jacobian, hessian, get_grad_concrete = get_grad_concrete)
        #print(np.linalg.eig(hessian))
        #import pdb; pdb.set_trace()
        x = x - tf.reshape(tf.linalg.solve(hessian, tf.reshape(jacobian, [4,1])),[4,]) # take Newton step
        residual_norm = residual_norm.numpy()

        print(f"Newton iteration {nits}.  Residual norm: {residual_norm}. X = {x.numpy()}")
        print(f"Newton  step: {tf.reshape(tf.linalg.solve(hessian, tf.reshape(jacobian, [4,1])),[4,])}")
        print(f"New x: {x}")
        if nits > 30:   break
        import pdb; pdb.set_trace()

    return x

if __name__ == "__main__":

    NGC_galaxies_results_dict = load_dict("NGC_galaxies_dict")
    for galaxy_name in NGC_galaxies_results_dict.keys():
        galaxy_dict = NGC_galaxies_results_dict[galaxy_name]
        s = get_s(galaxy_dict)
        pdb.set_trace()
