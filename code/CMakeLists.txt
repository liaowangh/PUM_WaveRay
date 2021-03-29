cmake_minimum_required(VERSION 3.10)

### Hunter package manager ###
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.288.tar.gz"
    SHA1 "6c9b2bc606d86ae31f96a62fc68f0a593024815b"
    LOCAL
)

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

# Get Boost
hunter_add_package(Boost COMPONENTS chrono timer system filesystem program_options)
find_package(Boost CONFIG REQUIRED chrono timer system filesystem program_options)
###############################################################################

# add_subdirectory(LagrangeO1)
# add_subdirectory(test)
# add_subdirectory(Pum)
# add_subdirectory(Pum_WaveRay)
# add_subdirectory(ExtendPum)
# add_subdirectory(ExtendPum_WaveRay)
add_subdirectory(Krylov)
# add_subdirectory(Wave_Ray_Cycle)
add_subdirectory(local_impedance)
