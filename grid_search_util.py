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
from matplotlib import cm
plt.rcParams.update({'font.size': 10})

def plot_grid_results(res_grid, x0_vals, x1_vals, x2_vals, best_loc):

    res_grid[np.where(np.isnan(res_grid))] = np.max(res_grid)
    X0, X1, X2 = np.meshgrid(x0_vals, x1_vals, x2_vals)
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 10})
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams.update({'font.size': 14})

    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(res_grid)
    img = ax.scatter(X0, X1, X2, c = np.minimum(res_grid/np.min(res_grid), 1.2*np.ones(res_grid.shape)), cmap = plt.cool(), alpha = 0.6)
    cax = ax.inset_axes([1.15, 0.2, 0.05, 0.6])
    cb = fig.colorbar(img, shrink=0.5, aspect=8, cax = cax)

    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('x2')
    ax.view_init(azim=-30, elev=20)
    fig.savefig("plots/residual_grid")

    #import pdb; pdb.set_trace()
    upper_bound = np.ones(res_grid.shape[0]) * 10**2 * np.min(res_grid)
    plt.clf()
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(131)
    plt.plot(x0_vals, np.minimum(res_grid[:,best_loc[1], best_loc[2]], upper_bound))
    plt.xlabel("x0"); plt.ylabel("residual")
    plt.yscale("log")

    plt.subplot(132)
    plt.plot(x1_vals, np.minimum(res_grid[best_loc[0], :, best_loc[2]], upper_bound))
    plt.xlabel("x1"); plt.ylabel("residual")
    plt.yscale("log")

    plt.subplot(133)
    plt.plot(x2_vals, np.minimum(res_grid[best_loc[0], best_loc[1], :], upper_bound))
    plt.xlabel("x2"); plt.ylabel("residual")
    plt.yscale("log")

    plt.savefig("res_vary_x.png")
    return

def grid_search(grid_ranges, galaxy_dict):
    num_samples = 17
    x0_vals = np.linspace(grid_ranges["x0"][0], grid_ranges["x0"][1], num_samples, endpoint = True)
    x1_vals = np.linspace(grid_ranges["x1"][0], grid_ranges["x1"][1], num_samples, endpoint = True)
    x2_vals = np.linspace(grid_ranges["x2"][0], grid_ranges["x2"][1], num_samples, endpoint = True)
    res_grid = np.empty((num_samples, num_samples, num_samples))
    res_grid.fill(np.nan)

    r, v, n, x = load_galaxy_data(galaxy_dict)
    get_grad_concrete = get_grad.get_concrete_function(r, v, n, x) # initialize tensorflow concrete function


    for i, x0 in enumerate(list(x0_vals)):
        for j, x1 in enumerate(list(x1_vals)):
            for k, x2 in enumerate(list(x2_vals)):
                x = tf.convert_to_tensor(np.asarray([x0, x1, x2]), dtype = "float64")
                residual_norm, jacobian, hessian, phi, s, v_calc = get_grad_concrete(r, v, n, x)
                res_grid[i, j, k] = residual_norm.numpy()

    res_grid[np.isnan(res_grid)] = np.Inf
    best_loc = np.unravel_index(np.argmin(res_grid.flatten()), res_grid.shape)
    best_x = tf.convert_to_tensor(np.asarray([x0_vals[best_loc[0]], x1_vals[best_loc[1]], x2_vals[best_loc[2]]]), dtype = "float64")

    tf.print("Best x from grid search: ", best_x)
    print(f"Lowest residual from grid search {np.min(res_grid)}")
    residual_norm, jacobian, hessian, phi, s, v_calc = get_grad_concrete(r, v, n, best_x)

    plot_grid_results(res_grid, x0_vals, x1_vals, x2_vals, best_loc)
    plot_velocities(r, v, v_calc, "grid_search")
    plot_s(r, best_x, "grid_search")

    import pdb; pdb.set_trace()
    return best_x



# plot grid_results
