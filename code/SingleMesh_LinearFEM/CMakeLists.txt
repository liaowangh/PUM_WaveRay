
set(SingleLinFEM_sources
  HE_LagrangeO1.h
  HE_LagrangeO1.cpp
  SingleLinFEM.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(SingleLinFEM ${SingleLinFEM_sources})

#link executable with LehrFEM++
target_link_libraries(SingleLinFEM
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
