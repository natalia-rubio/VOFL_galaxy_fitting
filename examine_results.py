import numpy as np
from scipy.special import gamma
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy
from os.path import exists
from tabulate import tabulate
from texttable import Texttable

font = {"size": 16}

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

NGC_galaxies_results_dict = load_dict("NGC_galaxies_results_dict")
rows = [["galaxy file", "x1", "x2", "s approximation error (L2 norm)"]]; x1 = []; x2 = []; fit_err = []

for galaxy_name in NGC_galaxies_results_dict.keys():
    galaxy_dict = NGC_galaxies_results_dict[galaxy_name]
    rows.append([galaxy_name, str(galaxy_dict["x1"]), str(galaxy_dict["x2"]), str(galaxy_dict["fit_err"])])
    x1.append(galaxy_dict["x1"]); x2.append(galaxy_dict["x2"]); fit_err.append(galaxy_dict["fit_err"])

# TABLE
table = Texttable()
table.set_cols_align(["c"] * 4)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('Tabulate Table:')
print(tabulate(rows, headers='firstrow'))

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))

# PLOT
plt.hist(x1, color = "cornflowerblue", label = "x1")
plt.hist(x2, color = "darksalmon", label = "x2")
plt.xlabel("coefficient value"); plt.ylabel("frequency")
plt.title("s Coefficient Fits in NGC Galaxies"); plt.legend()
plt.savefig("s_coefficient_fits")

pdb.set_trace()
