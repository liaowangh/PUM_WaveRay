set(mg_krylov_sources
  GMRES.h
  KrylovEnhanced_mg.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
  ../utils/utils.h
  ../utils/utils.cpp
)

# define executable
add_executable(mg_krylov ${mg_krylov_sources})

#link executable with LehrFEM++
target_link_libraries(mg_krylov
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

###################################################

# set(test_gmres_sources
#   GMRES.h
#   test_gmres.cpp
# )

# # define executable
# add_executable(test_gmres ${test_gmres_sources})

# #link executable with LehrFEM++
# target_link_libraries(test_gmres
#   PUBLIC Eigen3::Eigen
# )