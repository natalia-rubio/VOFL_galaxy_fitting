# VOFL_galaxy_fitting

---
*Goal: Fit order of fractional Laplacian in fractional Poisson equation for gravity to experimental data on orbital velocity versus radius for different galaxies.*

---

## Set up:

Assume the order of the fractional Laplacian, s(r) takes the form $s = \frac{x_0 + x_1 r}{x_2 + r} + x_3 $.  We want to find $\vec{x}  = [x_0, x_1, x_2, x_3]$.

We use the relation $v(s, r) = \sqrt{r \frac{d\Phi}{dr}}$ where r is radius, v is orbital velocity, and K is gravitational potential.  $\Phi$ is given by $G_N M \frac{ \Gamma \left(\frac{n}{2} - s(r) \right)}{4^{s (r)-1} \pi^{\frac{1}{2}} \Gamma \left(s (r) \right)} \frac{l^{2-2 s(r)}}{r^{n-2 s (r)}} $. 

We want to choose $s(r)$ such that the relationship between radius and orbital velocity is as close as possible to what we observe in experimental data.

We measure how far $s(r)$ is from capturing the true relationship for a set of radii and orbital velocities as $\epsilon = || \vec{v} (s, \vec{r}) - \vec{v}_{data} ||_2$.  The $\vec{x}$ that minimizes $\epsilon$ are that for which $\frac{d \epsilon}{d \vec{x}} = \vec{0}$.  To find this $\vec{x}$, we use Newton's Method where each Newton step is given as 

$$\vec{x}^{k+1} = \vec{x}^{k}- \left(\nabla^2 \epsilon(\vec{x}^{k})\right)^{-1} \nabla(\epsilon(\vec{x}^{k})).$$

## File Descriptions:
solve_for_s.py : Find $\vec{x}$ for each galaxy by running Newton's method described above on extracted orbital velocity vs radius data data. 

extract_galaxies.py : Extract orbital velocity vs radius data from .dat files downloaded from online database.

examine_results.py : Analyze distribution of $\vec{x}$ values for different galaxies.

grid_search_util.py: Use grid search to choose an initial guess for $\vec{x}$.

newton_util.py: Use Newton's method to fit $\vec{x}$ to data using tensorflow's automatic differentiation tools.

NGC_galaxies_dict : Dictionary storing extracted orbital velocity vs radius data for NGC galaxies.
