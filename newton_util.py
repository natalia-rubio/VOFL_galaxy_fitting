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
plt.rcParams.update({'font.size': 16})

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


def plot_newton_results(r, phi, s, v, v_calc, it_list, res_list):
    plt.clf(); plt.plot(r, phi.numpy())
    plt.xlabel("r (m)"); plt.ylabel("$\Phi$ ($m^2 s^{-2}$)")
    plt.savefig("plots/phi_vs_r_newton.png", bbox_inches = "tight")

    plt.clf(); plt.plot(r, s.numpy())
    plt.xlabel("r (m)"); plt.ylabel("$s$")
    plt.savefig("plots/s_vs_r_newton.png", bbox_inches = "tight")

    plt.clf()
    plt.plot(r, v, label = "data")
    plt.plot(r, v_calc, label = "predicted")
    plt.xlabel("r (m)"); plt.legend()
    plt.ylabel("orbital velocity ($m s^{-1}$)")
    plt.savefig("plots/r_vorb_newton.png", bbox_inches = "tight")

    plt.clf()
    plt.plot(it_list, res_list)
    plt.xlabel("iteration")
    plt.ylabel("residual_norm ($m s^{-1}$)")
    plt.yscale("log")
    plt.savefig("plots/residual_newton.png", bbox_inches = "tight")
    return

def run_newton_solve(initial_guess, galaxy_dict):
    r, v, n, x = load_galaxy_data(galaxy_dict)
    x = initial_guess
    res_list = []; it_list = []
    residual_norm = 1; tol = 0.1; nits = 0 # number of Newton iterations
    get_grad_concrete = get_grad.get_concrete_function(r, v, n, x) # initialize tensorflow concrete function
    print(f"First x: {x}")
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
            plot_newton_results(r, phi, s, v, v_calc, it_list, res_list)
            import pdb; pdb.set_trace()
            break
    print(f"x: {x}")
    print(f"residual_norm: {residual_norm}")
    plot_newton_results(r, phi, s, v, v_calc, it_list, res_list)
    return x
