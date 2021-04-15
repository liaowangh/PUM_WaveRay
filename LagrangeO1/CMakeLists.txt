# set(O1_resolution_sources
#   test/O1_resolution.cpp
#   HE_LagrangeO1.h
#   HE_LagrangeO1.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(O1_resolution ${O1_resolution_sources})

# #link executable with LehrFEM++
# target_link_libraries(O1_resolution
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

###################################################################################

set(multigrid_O1_sources
  test/multigrid_O1.cpp
  HE_LagrangeO1.h
  HE_LagrangeO1.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(multigrid_O1 ${multigrid_O1_sources})

#link executable with LehrFEM++
target_link_libraries(multigrid_O1
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

###################################################################################

# set(mode_analysis_sources
#   test/mode_analysis.cpp
#   HE_LagrangeO1.h
#   HE_LagrangeO1.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(mode_analysis ${mode_analysis_sources})

# #link executable with LehrFEM++
# target_link_libraries(mode_analysis
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

###################################################################################

set(O1_factor_sources
  test/O1_factor.cpp
  HE_LagrangeO1.h
  HE_LagrangeO1.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../utils/mg_element.h
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(O1_factor ${O1_factor_sources})

#link executable with LehrFEM++
target_link_libraries(O1_factor
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