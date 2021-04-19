# Class HE_ExtendPUM

This class inherits from the base class `HE_FEM`

```c++
class HE_ExtendPUM: virtual public HE_FEM
```

The `build_equation` generates the stiffness matrix and the right hand side vector built in Extend PUM spaces, and it depends on the function defined in 

- `ExtendPUM_EdgeMat.h` for generating local edge matrix

-  `ExtendPUM_EdgeVector.h` for generating local edge vector
- `ExtendPUM_ElementMatrix.h` for generating local element matrix
- `ExtendPUM_ElemVector.h` for generating local element vector

The file `tests/extendpum_resolution.cpp` does the resolution test for Extend PUM spaces,

it computes the norm of finite element solution obtained by direct solve in Extend PUM spaces minus the true solution. In current setting, the number of plane waves in Extend PUM spaces takes value in `3, 5, 7`,  wave number of Helmholtz equation takes the value `6, 20`. We can change the corresponding variable in main function for different wave number and number of plane waves.

