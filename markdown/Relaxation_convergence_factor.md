

## Domain: unit square

### Convergence factor of Lagrange Finite element operator, k  = 16

| k*h                | 16   | 8    | 4    | 2    | 1    | 0.5  |
| ------------------ | ---- | ---- | ---- | ---- | ---- | ---- |
| convergence factor | 0.37 | 0.77 | >>1  | 2.67 | 1.26 | 1.06 |

### Convergence factor of Extend PUM operator

| k*h                | 16   | 8    | 4    | 2    | 1    | 0.5  |
| ------------------ | ---- | ---- | ---- | ---- | ---- | ---- |
| convergence factor | >10  | >10  | >>1  | 2.62 | 1.26 | 1.06 |

### Convergence factor of  Extend PUM operator ($A_l=I_{l+1}^lA_{l+1}I_l^{l+1}$ )

| k*h                | 16          | 8          | 4           | 2       | 1      | 0.5     |
| ------------------ | ----------- | ---------- | ----------- | ------- | ------ | ------- |
| convergence factor | 2.05494e+13 | 1.6155e+14 | 4.07492e+07 | 16.5927 | 1.4546 | 1.05972 |



### Convergence factor of PUM-WaveRay operator

In finest mesh( $l=5, kh=0.5$), the operator is standard FEM operator (Galerkin matrix), for other mesh, 

the operator is the PUM operator.

| $l$  | $kh$ | factor(largest eigenvalue) | factor(power iteration) |
| ---- | ---- | -------------------------- | ----------------------- |
| 0    | 16   | 16.0065                    | 16.007                  |
| 1    | 8    | 27.46822                   | 27.4687                 |
| 2    | 4    | 19.0218                    | 19.0222                 |
| 3    | 2    | 2.38981                    | 2.38981                 |
| 4    | 1    | 1.26605                    | 1.26601                 |
| 5    | 0.5  | 1.05972                    | 1.05972                 |

### Convergence factor of PUM-WaveRay operator$A_l=I_{l+1}^lA_{l+1}I_l^{l+1}$ 

($I$ is computed based on best approximation of exponential function)

$k=16$

| $l$  | $kh$ | factor(largest eigenvalue) | factor(power iteration) |
| ---- | ---- | -------------------------- | ----------------------- |
| 0    | 16   | 3867.59                    | $\infty$                |
| 1    | 8    | 2.89567e+06                | $\infty$                |
| 2    | 4    | 8.74916e+10                | $\infty$                |
| 3    | 2    | 5.15762                    | 5.15974                 |
| 4    | 1    | 1.43515                    | 1.43505                 |
| 5    | 0.5  | 1.05972                    | 1.05959                 |

$k=2$

| $l$  | $kh$   | factor(largest eigenvalue) | factor(power iteration) |
| ---- | ------ | -------------------------- | ----------------------- |
| 0    | 2      | 3.08406                    | 3.08393                 |
| 1    | 1      | 1.19241                    | 1.19255                 |
| 2    | 0.5    | 1.0312                     | 1.03095                 |
| 3    | 0.25   | 1.00982                    | 1.00864                 |
| 4    | 0.125  | 1.00487                    | 1.00382                 |
| 5    | 0.0625 | 1.00036                    | 0.999368                |

$k=8$

| $l$  | $kh$ | factor(largest eigenvalue) | factor(power iteration) |
| ---- | ---- | -------------------------- | ----------------------- |
| 0    | 8    | 360500                     | 360502                  |
| 1    | 4    | 46926600                   | 46926600                |
| 2    | 2    | 6.10714                    | 6.04079                 |
| 3    | 1    | 1.20561                    | 1.20634                 |



## Domain: unit square with a hole

### Convergence factor of PUM-WaveRay operator$A_l=I_{l+1}^lA_{l+1}I_l^{l+1}$

k=8

| l    | kh       | factor(largest eigenvalue) | factor(power iteration) |
| ---- | -------- | -------------------------- | ----------------------- |
| 0    | 4.24264  | 1439.44                    | 1439.44                 |
| 1    | 2.12132  | 27.5214                    | 27.5213                 |
| 2    | 1.06066  | 1.41154                    | 1.41158                 |
| 3    | 0.53033  | 1.12465                    | 1.12413                 |
| 4    | 0.265165 | 1.00894                    | 1.00873                 |

k=16

| l    | kh      | factor(largest eigenvalue) | factor(power iteration) |
| ---- | ------- | -------------------------- | ----------------------- |
| 0    | 8.48528 | 8377.11                    | $\infty$                |
| 1    | 4.24264 | 4.83554e+09                | 4.83554e+09             |
| 2    | 2.12132 | 7799.97                    | 7799.97                 |
| 3    | 1.06066 | 1.33528                    | 1.33513                 |

### Convergence factor of Extend PUM-WaveRay operator$A_l=I_{l+1}^lA_{l+1}I_l^{l+1}$

k=8.0

| l    | kh       | factor(largest eigenvalue) | factor(power iteration) |
| ---- | -------- | -------------------------- | ----------------------- |
| 0    | 4.24264  | 987.956                    | 396.296                 |
| 1    | 2.12132  | 437.786                    | 33806.7                 |
| 2    | 1.06066  | 1.76626                    | 3.377                   |
| 3    | 0.53033  | 1.12764                    | 1.19379                 |
| 4    | 0.265165 | 1.00894                    | 1.00873                 |