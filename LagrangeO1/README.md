# Class LagrangeO1

This derived class inherits from the base class `HE_FEM`

```c++
class HE_LagrangeO1: virtual public HE_FEM
```

It rewrite the function `build_equation` to assemble the stiffness matrix and right hand side vector in Lagrangian finite element space.



In `test` folder

- `O1_factor.cpp` computes the convergence factor using the standard multigrid to solve the Helmholtz equation.
- `O1_resolution.cpp` use the direct solve to solve the Helmholtz equation with different wave number and different manufacture solution.