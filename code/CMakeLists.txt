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

# Get Boost
hunter_add_package(Boost COMPONENTS chrono timer system filesystem program_options)
find_package(Boost CONFIG REQUIRED chrono timer system filesystem program_options)

set(code_sources
  pum_wave_ray/pum_fem.h
  pum_wave_ray/pum_fem.cpp
  pum_wave_ray/main.cpp
)

#set(test_sources
#  test/test_solve_on_finest.cpp
#)

# Libraries
#add_library(pum_wave_ray.soulution ${code_sources})

# define executable
add_executable(pum_wave_ray ${code_sources})

#add_executable(pum_test ${test_sources})

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

#target_link_libraries(pum_test 
#  PUBLIC Eigen3::Eigen
#  LF::lf.mesh
#  LF::lf.mesh.utils
#  LF::lf.mesh.test_utils
#  LF::lf.mesh.hybrid2d
#  LF::lf.refinement
#  LF::lf.assemble
#  LF::lf.io
#  LF::lf.uscalfe
#  LF::lf.base
#  GTest::Main
#  pum_wave_ray.solution
#)

#gtest_discover_tests(pum_test)
