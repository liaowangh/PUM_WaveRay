cmake_minimum_required(VERSION 3.10)

### Hunter package manager ###
include("cmake/HunterGate.cmake")

option(HUNTER_STATUS_DEBUG "Print a lot of info" ON)

HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.249.tar.gz"
    SHA1 "d45d77d8bba9da13e9290a180e0477e90accd89b"
)

### Setup CMake Project ###
project(test_gmsh)

# download dependencies
hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

# define executable
#add_executable(test_gmsh test_gmsh.cpp)
add_executable(test_point_parent test_point_parent.cpp)

#link executable with LehrFEM++
target_link_libraries(test_point_parent
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
