
set(test_LagrangeO1_sources
  HE_LagrangeO1.h
  HE_LagrangeO1.cpp
  test_LagrangeO1.cpp
  ../pum_wave_ray/HE_FEM.h
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(test_LagrangeO1 ${test_LagrangeO1_sources})

#link executable with LehrFEM++
target_link_libraries(test_LagrangeO1
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
