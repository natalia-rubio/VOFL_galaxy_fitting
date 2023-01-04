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

@tf.function
def get_grad(r, v, n, x):

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

                # mass 10^39
                K_fac = tf.constant(6.67*10**(-11) * 10**(40), dtype = tf.float64)
                phi = tf.multiply(K_fac, tf.divide(num, den)) # get K - gravitational potential

            dphidr = tape_r.gradient(phi, r)
            #v_calc = tf.sqrt(tf.multiply(tf.multiply(tf.convert_to_tensor(-1, dtype = "float64"), dphidr), r))
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

def plot_velocities(r, v_data, v_calc, plot_name):
    plt.clf()
    plt.plot(r, v_data, label = "data")
    plt.plot(r, v_calc, label = "predicted")
    plt.xlabel("r (m)"); plt.legend()
    plt.ylabel("orbital velocity ($m s^{-1}$)")
    plt.savefig(f"plots/r_vorb_{plot_name}.png", bbox_inches = "tight")
    return

def plot_v2r(r, v_data, plot_name):
    plt.clf()
    plt.plot(r, np.square(v_data)/r, label = "data")
    plt.xlabel("r (m)"); plt.legend()
    plt.ylabel("centripetal acceleration ($v^{-2}r^{-1}$)")
    plt.savefig(f"plots/cent_acc_{plot_name}.png", bbox_inches = "tight")
    return

def plot_s(r, x, plot_name):
    r = r.numpy()
    x = x.numpy()
    #import pdb; pdb.set_trace()
    s = (x[0] + x[1]*r)/(x[2] + r)
    #print(f"s: {s}")

    plt.clf(); plt.plot(r, s)
    plt.xlabel("r (m)"); plt.ylabel("$s$")
    plt.savefig(f"plots/s_vs_r_{plot_name}.png", bbox_inches = "tight")
    return

def plot_phi(r, phi, plot_name):
    plt.clf(); plt.plot(r, phi.numpy())
    plt.xlabel("r (m)"); plt.ylabel("$\Phi$ ($m^2 s^{-2}$)")
    plt.savefig(f"plots/phi_{plot_name}.png", bbox_inches = "tight")
