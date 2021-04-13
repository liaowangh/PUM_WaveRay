## Local smoothing with impedance boundary conditions

**Multi-grid smoothing strategy:**

- For $kh<1.0$, Gauss-Seidel 
- For $kh\ge1.0$: smoothing with impedance boundary conditions

Then use the multi-grid method as a preconditioner  for GMRES,

execute $m$ steps of GMRES, then restart with $u_m$ as initial approximation of the next GMRES cycle.



### Domain: unit square, $k = 15$, $h_L=1/16$

| $i$       | 0    | 1    | 2    | 3    | 4        | 5        | 6        |
| --------- | ---- | ---- | ---- | ---- | -------- | -------- | -------- |
| $||e_i||$ | 1.13 | 0.61 | 0.30 | 0.12 | $4.0e-2$ | $1.4e-2$ | $6.3e-3$ |



### Domain: unit square with a hole, $k = 15$, $h_L=1/16$

| $i$       | 0    | 1    | 2        | 3        | 4        | 5        | 6        |
| --------- | ---- | ---- | -------- | -------- | -------- | -------- | -------- |
| $||e_i||$ | 1.03 | 0.29 | $6.8e-2$ | $1.9e-2$ | $3.7e-3$ | $8.7e-4$ | $1.6e-4$ |



### Domain: unit square with a triangle hole, $k = 15$, $h_L=1/16$

| $i$       | 0    | 1    | 2    | 3    | 4        | 5        | 6        |
| --------- | ---- | ---- | ---- | ---- | -------- | -------- | -------- |
| $||e_i||$ | 1.07 | 0.49 | 0.25 | 0.11 | $4.6e-2$ | $2.0e-2$ | $8.7e-3$ |