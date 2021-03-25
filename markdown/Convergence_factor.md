# Convergence factor of two-mesh correction scheme

## Domain: unit square

| $k*h$       | convergence factor |
| ----------- | ------------------ |
| 0.125~3.375 | $<1.0e-5$          |
| 3.5         | 0.48               |
| 3.625       | 79.7               |
| 3.75        | $3.4e+06$          |
| 3.875       | $2.1e+17$          |
| 4           | $\infty$           |
| 4.125       | $3.3e+15$          |
| 4.25        | $3.5e+03$          |
| 4.375       | 0.1                |
| 4.5~8       | $ < 1.0e-05$       |



**For $kh=4$, there are a lot of zeros in the diagonal of the Galerkin matrix**

### Convergence factor of G-S iteration of Extend PUM operator ($A_l=I_{l+1}^lA_{l+1}I_l^{l+1}$ )

| k*h                | 16        | 8         | 4         | 2     | 1    | 0.5  |
| ------------------ | --------- | --------- | --------- | ----- | ---- | ---- |
| convergence factor | $2.1e+13$ | $1.6e+14$ | $4.1e+07$ | 16.60 | 1.45 | 1.06 |



## Domain: unit square with a square hole

| $k*h$       | convergence factor |
| ----------- | ------------------ |
| $<5.25$     | $<1.0e-6$          |
| 5.375       | 0.54               |
| 5.5         | 8.0                |
| 5.625~6.875 | div                |
| 7~7.625     | $<0.1$             |
| 7.75~8      | div                |



## Domain: unit square with a triangle hole

| $k*h$    | convergence factor |
| -------- | ------------------ |
| $<4.125$ | $<2.0e-4$          |
| 4.25-8   | div                |



# Convergence factor of three-mesh correction scheme

## Domain: unit square

| $k*h$       | convergence factor |
| ----------- | ------------------ |
| 0.125~0.875 | $<1.0e-3$          |
| 1           | 0.08               |
| 1.125       | 0.17               |
| 1.25        | 0.35               |
| 1.375       | 0.75               |
| 1.5         | 1.6                |
| 1.75        | 9.0                |
| 1.875       | 22.6               |
| 2           | 59                 |
| 2.25~5.5    | div                |



## Domain: unit square with a square hole

| $k*h$       | convergence factor |
| ----------- | ------------------ |
| 0.125~1.375 | $<0.01$            |
| 1.5~1.875   | $<0.5$             |
| 2           | 0.65               |
| 2.125       | 1.04               |
| 2.25        | 2.1                |
| 2.375       | 5.67               |
| 2.5~3.5     | 10~1000            |
| 3.625~4.5   | 1000~$1.0e+6$      |
| 4.625~7     | $>1.0e+06$         |



## Domain: unit square with a triangle hole

| $k*h$       | convergence factor |
| ----------- | ------------------ |
| 0.125~1.375 | $\approx 0.10$     |
| 1.5         | 0.24               |
| 1.625       | 0.44               |
| 1.75        | 0.84               |
| 1.875       | 1.64               |
| 2           | 3.30               |
| 2.125~7     | $>1$ , increasing  |
|             |                    |
|             |                    |