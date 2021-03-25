# set(test_impedance_sources
#   test_impedance.cpp
#   local_impedance_solver.h
#   ../LagrangeO1/HE_LagrangeO1.h
#   ../LagrangeO1/HE_LagrangeO1.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(test_impedance ${test_impedance_sources})

# #link executable with LehrFEM++
# target_link_libraries(test_impedance
#   PUBLIC Eigen3::Eigen Boost::filesystem Boost::system Boost::program_options
#   LF::lf.mesh
#   LF::lf.mesh.utils
#   LF::lf.mesh.test_utils
#   LF::lf.mesh.hybrid2d
#   LF::lf.refinement
#   LF::lf.assemble
#   LF::lf.io
#   LF::lf.uscalfe
#   LF::lf.base
# )

#############################################################################

set(local_impedance_smoothing_sources
  local_impedance_smoothing.cpp
  local_impedance_solver.h
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../Krylov/GMRES.h
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(local_impedance_smoothing ${local_impedance_smoothing_sources})

#link executable with LehrFEM++
target_link_libraries(local_impedance_smoothing
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