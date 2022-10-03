# VOFL_galaxy_fitting

---
*Goal: Fit order of fractional Laplacian in fractional Poisson equation for gravity to experimental data on orbital velocity versus radius for different galaxies.*

---

## Set up:

Assume the order of the fractional Laplacian, s(r) takes the form $s = \frac{x_0 + x_1 r}{x_2 + r} + x_3 $.  We want to find $\vec{x}  = [x_0, x_1, x_2, x_3]$.

We use the relation $v(s, r) = \sqrt{r \frac{dK}{dr}}$ where r is radius, v is orbital velocity, and K is gravitational potential.  $K$ is given by $- \frac{ \Gamma \left(\frac{n}{2} - s(r) \right)}{4^{s (r)} \pi^{\frac{n}{2}} \Gamma \left(s (r) \right)} \frac{1}{r^{n-2 s (r)}} $.  (Note: add in constants from Andrea)

We want to choose $s(r)$ such that the relationship between radius and orbital velocity is as close as possible to what we observe in experimental data.

We measure how far $s(r)$ is from capturing the true relationship for a set of radii and orbital velocities as $\epsilon = || \vec{v} (s, \vec{r}) - \vec{v}_{data} ||_2$.  (Assuming convexity?  Should check this -->) The $\vec{x}$ that minimizes $\epsilon$ are that for which $\frac{d \epsilon}{d \vec{x}} = \vec{0}$.  To find this $\vec{x}$, we use Newton's Method where each Newton step is given as 

$$\vec{x}^{k+1} = \vec{x}^{k}- \left(\nabla^2 \epsilon(\vec{x}^{k})\right)^{-1} \nabla(\epsilon(\vec{x}^{k})).$$

## File Descriptions:
solve_for_s.py : Find $\vec{x}$ for each galaxy by running Newton's method described above on extracted orbital velocity vs radius data data. [CURRENTLY NOT CONVERGING]

extract_galaxies.py : Extract orbital velocity vs radius data from .dat files downloaded from online database.

examine_results : Analyze distribution of $\vec{x}$ values for different galaxies.  (Not relevant until solver is working.)

NGC_galaxies_dict : Dictionary storing extracted orbital velocity vs radius data for NGC galaxies.
