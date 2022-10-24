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
plt.rcParams.update({'font.size': 16})

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def load_galaxy_data(galaxy_dict):
    r = tf.constant(galaxy_dict["r"].astype(np.float64)) # load in radius
    v = tf.constant(galaxy_dict["v"].astype(np.float64)) # load in orbital velocity
    n = tf.constant(np.asarray([3.0,]).astype(np.float64)) # set number of dimensions
    x = np.asarray([0.001, 1.5, 0.01]) # initial guesses for x
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

                s = tf.math.divide(tf.math.add(tf.math.multiply(x[0], \
                tf.math.pow(tf.convert_to_tensor(10, dtype = "float64"), tf.convert_to_tensor(20 , dtype = "float64"))),
                tf.math.multiply(x[1], r)),
                tf.math.add(tf.math.multiply(x[2], \
                tf.math.pow(tf.convert_to_tensor(10, dtype = "float64"), tf.convert_to_tensor(20 , dtype = "float64"))), r))

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
                num = tf.multiply(gamma_num, length_scale)

                K_fac = tf.constant(-6.67*10**(-11) * 10**(39), dtype = tf.float64)
                phi = tf.multiply(K_fac, tf.divide(num, den)) # get K - gravitational potential

            dphidr = tf.math.abs(tape_r.gradient(phi, r))
            v_calc = tf.sqrt(tf.multiply(dphidr, r))
            if False:
                tf.print("num", num)
                tf.print("den", den)
                tf.print("dnum dr", tape_r.gradient(num, r))
                tf.print("dlden dr", tape_r.gradient(den, r))
                tf.print("quotient first term", tf.multiply(den, tape_r.gradient(num, r)))
                tf.print("quotient second term", tf.multiply(num, tape_r.gradient(den, r)))
                tf.print("d gamma den dr", tape_r.gradient(gamma_den, r))
                tf.print("d abs den dr", tape_r.gradient(abs_den, r))

                tf.print("s: ", s)
                tf.print("Phi: ", phi)
                tf.print("max Phi: ", tf.math.reduce_max(phi))
                tf.print("r shape: ", tf.shape(r))
                tf.print("dPhidr: ", dphidr)
                tf.print("s: ", s)
                tf.print("length scale: ", length_scale)
                tf.print("r: ", r)
                tf.print("r * dPhidr: ", tf.multiply(r, dphidr))
                tf.print("dPhidr shape: ", tf.shape(dphidr))
                tf.print("dPhidr maximum: ", tf.reduce_max(dphidr))

                tf.print("dPhidr", dphidr, summarize = 28)
                tf.print("sqrt r * dPhidr", tf.sqrt(tf.multiply(dphidr, r)))
                tf.print("before norm ", tf.subtract( \
                tf.sqrt(tf.multiply(dphidr, r)), tf.math.abs(v)))
                tf.print("V :", v)

            residual_norm = tf.math.reduce_euclidean_norm(tf.subtract( \
            tf.sqrt(tf.multiply(dphidr, r)), tf.math.abs(v)))

        jacobian = tape_jac.jacobian(residual_norm, x)

    hessian = tape_hess.jacobian(jacobian, x)

    return residual_norm, jacobian, hessian, phi, s, v_calc

def get_s(galaxy_dict):
    res_list = []
    it_list = []
    r, v, n, x = load_galaxy_data(galaxy_dict)
    residual_norm = 1
    tol = 0.1 # acceptable residual norm
    nits = 0 # number of Newton iterations
    get_grad_concrete = get_grad.get_concrete_function(r, v, n, x) # initialize tensorflow concrete function
    while residual_norm > tol:
        nits += 1
        residual_norm, jacobian, hessian, phi, s, v_calc = get_grad_concrete(r, v, n, x) # tf function to get res_norm, d res_norm/dx, and d^2 res_norm/dx^2

        try:
            x = x - tf.reshape(tf.linalg.solve(hessian, tf.reshape(jacobian, [3,1])),[3,]) # take Newton step
        except:
            import pdb; pdb.set_trace()

        residual_norm = residual_norm.numpy()

        print(f"Newton iteration {nits}.  Residual norm: {residual_norm}. X = {x.numpy()}")
        print(f"Newton  step: {tf.reshape(tf.linalg.solve(hessian, tf.reshape(jacobian, [3,1])),[3,])}")
        print(f"New x: {x}")
        it_list.append(nits); res_list.append(residual_norm)
        if nits == 60:
            plt.clf(); plt.plot(r, phi.numpy())
            plt.xlabel("r (m)"); plt.ylabel("$\Phi$ ($m^2 s^{-2}$)")
            plt.savefig("phi_vs_r.png", bbox_inches = "tight")

            plt.clf(); plt.plot(r, s.numpy())
            plt.xlabel("r (m)"); plt.ylabel("$s$")
            plt.savefig("s_vs_r.png", bbox_inches = "tight")

            plt.clf()
            plt.plot(r, v, label = "data")
            plt.plot(r, v_calc, label = "predicted")
            plt.xlabel("r (m)"); plt.legend()
            plt.ylabel("orbital velocity ($m s^{-1}$)")
            plt.savefig("r_vorb.png", bbox_inches = "tight")

            plt.clf()
            plt.plot(it_list, res_list)
            plt.xlabel("iteration")
            plt.ylabel("residual_norm ($m s^{-1}$)")
            plt.yscale("log")
            plt.savefig("residual.png", bbox_inches = "tight")
            import pdb; pdb.set_trace()
            break
    return x

if __name__ == "__main__":

    NGC_galaxies_results_dict = load_dict("NGC_galaxies_dict")
    for galaxy_name in NGC_galaxies_results_dict.keys():
        galaxy_dict = NGC_galaxies_results_dict[galaxy_name]
        s = get_s(galaxy_dict)
        pdb.set_trace()
