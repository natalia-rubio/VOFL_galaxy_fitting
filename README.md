# VOFL_galaxy_fitting

---
*Goal: Fit order of fractional Laplacian in fractional Poisson equation for gravity to experimental data on orbital velocity versus radius for different galaxies.*

---

## Set up:

Assume the order of the fractional Laplacian, s(r) takes the form $s = \frac{x_0 + x_1 r}{x_2 + r} + x_3 $.  We want to find $x_0, x_1, x_2, x_3$.

We use the relation $v_{analytical}(s, r) = \sqrt{r \frac{dK}{dr}}$ where r is radius, v is orbital velocity, and K is gravitational potential.  $K$ is given by $- \frac{ \Gamma \left(\frac{n}{2} - s(r) \right)}{4^{s (r)} \pi^{\frac{n}{2}} \Gamma \left(s (r) \right)} \frac{1}{r^{n-2 s (r)}} $.  (Note: add in constants from Andrea)

We want to choose $s(r)$ such that the relationship between radius and orbital velocity is as close as possible to what we observe in experimental data.

We measure how far $s(r)$ is from capturing the true relationship for a set of radii and orbital velocities as $|| \vec{v}_{analytical} (s, \vec{r}) - \vec{v}_{data} ||_2 $.
