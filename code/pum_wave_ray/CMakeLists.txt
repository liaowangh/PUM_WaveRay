cmake_minimum_required(VERSION 3.10)

### Hunter package manager ###
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.249.tar.gz"
    SHA1 "d45d77d8bba9da13e9290a180e0477e90accd89b"
)

### Setup CMake Project ###
project(GettingStarted)

# download dependencies
hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

# define executable
add_executable(pum_wave_ray main.cpp)

#link executable with LehrFEM++
target_link_libraries(pum_wave_ray 
  PUBLIC Eigen3::Eigen
  LF::lf.mesh
  LF::lf.mesh.utils
  LF::lf.mesh.test_utils
  LF::lf.mesh.hybrid2d
  LF::lf.refinement
  LF::lf.assemble
  LF::lf.io
  LF::lf.uscalfe
  LF::lf.base
)
