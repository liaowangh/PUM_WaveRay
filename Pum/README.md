# Class HE_PUM

This class inherits from the base class `HE_FEM`

```c++
class HE_PUM: virtual public HE_FEM
```

The `build_equation` generates the stiffness matrix and the right hand side vector built in PUM space, and it depends on the function defined in 

- `PUM_EdgeMat.h` for generating local edge matrix
-  `PUM_EdgeVector.h` for generating local edge vector
- `PUM_ElementMatrix.h` for generating local element matrix
- `PUM_ElemVector.h` for generating local element vector

The file `tests/pum_resolution.cpp` does the resolution test for PUM spaces,

it computes the norm of finite element solution obtained by direct solve in PUM spaces minus the true solution. In current setting, the number of plane waves in PUM spaces takes value in `3, 5, 7, 8, 11, 13`,  wave number of Helmholtz equation takes the value `6, 20, 60`. We can change the corresponding variable in main function for different wave number and number of plane waves.