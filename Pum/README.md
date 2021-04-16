# Class HE_PUM

The derived class

```c++
class HE_PUM: virtual public HE_FEM
```

The `build_equation` function in it generates the stiff matrix and right hand side vector built at PUM space, and it depends on the function defined in 

- `PUM_EdgeMat.h` for generating local edge matrix
-  `PUM_EdgeVector.h` for generating local edge vector
- `PUM_ElementMatrix.h` for generating local element matrix
- `PUM_ElemVector.h` for generating local element vector

The file `tests/pum_resolution.cpp` does the resolution test for PUM spaces,

it computes the error norm of finite element solution obtained by direct solve in PUM spaces and the true solution.

