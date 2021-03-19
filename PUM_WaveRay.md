# PUM Wave-Ray

First leg of wave cycle: from finest level 5 ($h=0.03125$) to coarsest level 0 ($h = 1$),

Second leg of wave cycle: from coarsest to level  $L_0$ , then transfer the residual to Extend PUM space to perform the ray cycle.

Finish the second leg of wave cycle.

|    $l$     | 5     | 4    | 3    | 2    | 1    | 0      |
| :--------: | ----- | ---- | ---- | ---- | ---- | ------ |
|   $k*h$    | 0.625 | 1.25 | 2.5  | 5    | 10   | 20     |
| Relaxation | G-S   | G-S  | None | None | G-S  | solved |

$L_0=4$, use 1 Extend PUM space. 

|  $l$  | 5     | 4                                          | 3      | 2    | 1    | 0    |
| :---: | ----- | ------------------------------------------ | ------ | ---- | ---- | ---- |
| $k*h$ | 0.625 | 1.25                                       | 2.5    | 5    | 10   | 20   |
|       |       | transfer residual to next Extend PUM space | solved |      |      |      |

Let $u_h$ be the finite element solution, $v_i$ the approximate solution after $i$th iteration, $e_i=u_h-v_i$ be the error function after $i$th iteration.

### Domain: unit square with a square hole, $k = 20$

| $i$       | 0    | 1    | 2    | 3        | 4        | 5        | 6        |
| --------- | ---- | ---- | ---- | -------- | -------- | -------- | -------- |
| $||e_i||$ | 1.03 | 0.52 | 0.15 | $4.4e-2$ | $1.5e-2$ | $5.8e-3$ | $2.3e-3$ |

  convergence factor $\approx 0.4$.

### Domain: unit square, $k = 20$

| $i$       | 0    | 1    | 2        | 3        | 4        | 5        | 6        |
| --------- | ---- | ---- | -------- | -------- | -------- | -------- | -------- |
| $||e_i||$ | 1.13 | 0.28 | $6.1e-2$ | $1.3e-2$ | $2.6e-3$ | $4.8e-4$ | $8.8e-5$ |

  convergence factor $\approx 0.2$.

### Domain: unit square with triangle hole, $k = 20$

| $i$       | 0    | 1    | 2        | 3        | 4        | 5        | 6        |
| --------- | ---- | ---- | -------- | -------- | -------- | -------- | -------- |
| $||e_i||$ | 1.06 | 0.24 | $6.8e-2$ | $2.4e-2$ | $9.9e-3$ | $4.6e-3$ | $2.2e-3$ |

  convergence factor $\approx 0.45$.