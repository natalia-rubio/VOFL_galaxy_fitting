import numpy as np
from scipy.special import gamma
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy
from os.path import exists
font = {"size": 16}

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_galaxy_data(galaxy_name):

    galaxy_data = np.char.split(np.genfromtxt('galaxy_data/rot_mod/'+galaxy_name,
                        encoding=None,
                        skip_header=1,
                        skip_footer=1,
                        names=True,
                        dtype=None,
                        delimiter='  ').astype(str), "\t")

    r = np.zeros((galaxy_data.size,))
    v = np.zeros((galaxy_data.size,))
    for i in range(galaxy_data.size):
        r[i] = float(galaxy_data[i][0])
        v[i] = float(galaxy_data[i][1])
    galaxy_dict = {"r": np.asarray(r).astype(np.longdouble)*3.086e+19, # kpc to meters
                    "v": np.asarray(v).astype(np.longdouble)*1000 } # km s^-1 to m s^-1

    return galaxy_dict

def extract_galaxies():

    with open("galaxy_data/rot_mod/filelist.txt") as f:
        content = f.readlines() # load in models from repository
    galaxy_names = [x.strip() for x in content].copy()

    NGC_galaxies_dict = {}
    for galaxy_name in galaxy_names:
        if galaxy_name[0:3] == "NGC":
            galaxy_dict = get_galaxy_data(galaxy_name)
            NGC_galaxies_dict.update({galaxy_name: galaxy_dict})

    save_dict(NGC_galaxies_dict, "NGC_galaxies_dict")

    return

if __name__ == "__main__":
    extract_galaxies()
