# PUM-Based Wave-Ray Multigrid

This repository contains the codes for the master thesis project `PUM-Based Wave-Ray multigrid`, the results of numerical experiments can be found at 

`document/Master_Thesis_Liaowang.pdf`

This markdown aims to show how to repeat the results of the numerical experiments. Details can be found at the markdown file in the subdirectory.

At first, cmake with the release mode

```bash
mkdir release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

- Resolution test for standard Lagrangian finite elmeent spaces
  - Results are in section 7.1 in the thesis pdf.
  - Source file: `LagrangeO1/test/O1_resolution.cpp`
  - Run the executable file `./LagrangeO1/O1_resolution`
  - The output is stored at `result_squarehole/LagrangeO1/`
- Resolution test for (Extend) PUM spaces
  - Results are in section 5.4 in the thesis pdf.
  - Source files lie in `Pum/tests/pum_resolution.cpp`  and `ExtendPum/tests/extendpum_resolution.cpp`
  - Run the executable file `./Pum/pum_resolution` and `./ExtendPum/ExtendPum_resolution`
  - The output is stored at `result_square/PUM` and `result_square/ExtendPum/`
- PUM Wave-Ray method
  - Results are in section 7.3 
  - Source file lies in `Wave_Ray_Cycle/Wave_Ray.cpp`, and some useful functions are in `utils/mg_element.h`
  - Run the executable `./Wave_Ray_Cycle/Wave_Ray`
  - The output is stored at `result/waveray_factor/`

- GMRES

  - Results are in section 8.3
  - Source file: `Krylov/KrylovEnhance_mg.cpp`
  - Run the executable file `./Krylov/mg_krylov`
  - The output is stored at `result/mgGmres_count/`

- Local impedance smoothing

  - Results in section 8.3

  - Files: `local_impedance/local_impedance_smoothing.cpp`

  - Run the executable file `./local_impedance/local_impedance_smoothing`
  - The output is stored at `result/impedance_count`

    











