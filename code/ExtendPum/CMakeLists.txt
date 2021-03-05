set(ExtendPum_resolution_sources
  tests/extendpum_resolution.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ExtendPUM_EdgeMat.h
  ExtendPUM_EdgeMat.cpp
  ExtendPUM_EdgeVector.h
  ExtendPUM_EdgeVector.cpp
  ExtendPUM_ElementMatrix.h
  ExtendPUM_ElementMatrix.cpp
  ExtendPUM_ElemVector.h
  ExtendPUM_ElemVector.cpp
  HE_ExtendPUM.h
  HE_ExtendPUM.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(ExtendPum_resolution ${ExtendPum_resolution_sources})

#link executable with LehrFEM++
target_link_libraries(ExtendPum_resolution
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

###############################################################################################

# set(BoundaryProj_sources
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ExtendPUM_EdgeMat.h
#   ExtendPUM_EdgeMat.cpp
#   ExtendPUM_EdgeVector.h
#   ExtendPUM_EdgeVector.cpp
#   ExtendPUM_ElementMatrix.h
#   ExtendPUM_ElementMatrix.cpp
#   ExtendPUM_ElemVector.h
#   ExtendPUM_ElemVector.cpp
#   HE_ExtendPUM.h
#   HE_ExtendPUM.cpp
#   tests/boundary_projection.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(BoundaryProj ${BoundaryProj_sources})

# #link executable with LehrFEM++
# target_link_libraries(BoundaryProj
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

###############################################################################################

# set(mg_extendPum_sources
#   tests/multigrid_extendPum.cpp
#   ExtendPUM_EdgeMat.h
#   ExtendPUM_EdgeMat.cpp
#   ExtendPUM_EdgeVector.h
#   ExtendPUM_EdgeVector.cpp
#   ExtendPUM_ElementMatrix.h
#   ExtendPUM_ElementMatrix.cpp
#   ExtendPUM_ElemVector.h
#   ExtendPUM_ElemVector.cpp
#   HE_ExtendPUM.h
#   HE_ExtendPUM.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(mg_extendPum ${mg_extendPum_sources})

# #link executable with LehrFEM++
# target_link_libraries(mg_extendPum
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

###############################################################################################

# set(test_extendPum_sources
#   tests/test_extendPum.cpp
#   ExtendPUM_EdgeMat.h
#   ExtendPUM_EdgeMat.cpp
#   ExtendPUM_EdgeVector.h
#   ExtendPUM_EdgeVector.cpp
#   ExtendPUM_ElementMatrix.h
#   ExtendPUM_ElementMatrix.cpp
#   ExtendPUM_ElemVector.h
#   ExtendPUM_ElemVector.cpp
#   HE_ExtendPUM.h
#   HE_ExtendPUM.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(test_extendPum ${test_extendPum_sources})

# #link executable with LehrFEM++
# target_link_libraries(test_extendPum
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