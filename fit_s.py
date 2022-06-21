from extract_s import *
from sklearn.linear_model import LinearRegression

NGC_galaxies_results_dict = load_dict("NGC_galaxies_dict")

def lin_reg_s(r, s):

    lin_reg = LinearRegression()
    s_terms = np.hstack(((1/(1+r)).reshape(-1,1), (r/(1+r)).reshape(-1,1)))
    lin_reg.fit(s_terms.astype(np.longdouble), s.astype(np.longdouble))
    x1 = lin_reg.coef_[0]; x2 = lin_reg.coef_[1]
    fit_err = np.linalg.norm(lin_reg.predict(s_terms) - s)

    return x1, x2, fit_err

for galaxy_name in NGC_galaxies_results_dict.keys():
    galaxy_dict = NGC_galaxies_results_dict[galaxy_name]
    s = get_s(galaxy_dict)
    x1, x2, fit_err = lin_reg_s(galaxy_dict["r"], s)
    galaxy_dict.update({"s": s,
                        "x1": x1,
                        "x2": x2,
                        "fit_err": fit_err})
    NGC_galaxies_results_dict.update({galaxy_name: galaxy_dict})

save_dict(NGC_galaxies_results_dict, "NGC_galaxies_results_dict")
