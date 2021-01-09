cmake_minimum_required(VERSION 3.10)

### Hunter package manager ###
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.288.tar.gz"
    SHA1 "6c9b2bc606d86ae31f96a62fc68f0a593024815b"
    LOCAL
)

include(GoogleTest)

### Setup CMake Project ###
project(PUM_Wave_Ray)

# set compiler
set(CMAKE_CXX_STANDARD 17)

# Get Dependencies
###############################################################################
hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

# Get Eigen
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

# Get Google Test
hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)

# Get Boost
hunter_add_package(Boost COMPONENTS chrono timer system filesystem program_options)
find_package(Boost CONFIG REQUIRED chrono timer system filesystem program_options)
###############################################################################

set(code_sources
  pum_wave_ray/pum_fem.h
  pum_wave_ray/pum_fem.cpp
  pum_wave_ray/main.cpp
  utils/utils.h
  utils/utils.cpp
  utils/HE_solution.h
  utils/HE_solution.cpp
)

set(Laplace_sources
  pum_wave_ray/pum_fem.h
  pum_wave_ray/pum_fem.cpp
  pum_wave_ray/main_Laplace.cpp
  utils/utils.h
  utils/utils.cpp
  utils/HE_solution.h
  utils/HE_solution.cpp
)

# define executable
add_executable(pum_wave_ray ${code_sources})

#add_executable(Laplace ${Laplace_sources})

#link executable with LehrFEM++
target_link_libraries(pum_wave_ray
  PUBLIC Eigen3::Eigen Boost::filesystem Boost::system Boost::program_options
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
