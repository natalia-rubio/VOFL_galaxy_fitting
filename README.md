# VOFL_galaxy_fitting

---
*Goal: Fit order of fractional Laplacian to experimental data on orbital velocity versus radius for different galaxies.*

---

## Set up:

Assume s(r) takes the form $s = \frac{x_0 + x_1 r}{x_2 + r} + x_3 $.  We want to find $x_0, x_1, x_2, x_3$.

We use the relation $\frac{dK}{dr} = \frac{v^2}{r}$ where r is radius, v is orbital velocity, and K is gravitational potential.  

$K$ is given by $- \frac{ \Gamma \left(\frac{n}{2} - s(r) \right)}{4^{s (r)} \pi^{\frac{n}{2}} \Gamma \left(s (r) \right)} \frac{1}{r^{n-2 s (r)}} $
