cmake_minimum_required(VERSION 3.10)

### Hunter package manager ###
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.249.tar.gz"
    SHA1 "d45d77d8bba9da13e9290a180e0477e90accd89b"
)

include(GoogleTest)

### Setup CMake Project ###
project(PUM_Wave_Ray)

# download dependencies
hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

# Get Eigen
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

# Get Google Test
hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)

add_subdirectory(test)
add_subdirectory(pum_wave_ray)


set(pum_code, 
  pum_wave_ray/pum_fem.h
  pum_wave_ray/pum_fem.cpp
)

set(pum_test,
  test/test_solve_on_finest.cpp
)

# define executable
#add_executable(pum_wave_ray ${pum_code})

add_executable(pum_test ${pum_test})

#link executable with LehrFEM++
target_link_libraries(pum_test 
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
