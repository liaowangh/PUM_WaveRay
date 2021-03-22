set(Wave_Ray_sources
  Wave_Ray.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../Pum_WaveRay/PUM_WaveRay.h
  ../Pum_WaveRay/PUM_WaveRay.cpp
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../Pum/PUM_EdgeMat.h
  ../Pum/PUM_EdgeVector.h
  ../Pum/PUM_ElementMatrix.h
  ../Pum/PUM_ElemVector.h
  ../Pum/HE_PUM.h
  ../Pum/HE_PUM.cpp
  ../ExtendPum_WaveRay/ExtendPUM_WaveRay.h
  ../ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp
  ../ExtendPum/ExtendPUM_EdgeMat.h
  ../ExtendPum/ExtendPUM_EdgeVector.h
  ../ExtendPum/ExtendPUM_ElementMatrix.h
  ../ExtendPum/ExtendPUM_ElemVector.h
  ../ExtendPum/HE_ExtendPUM.h
  ../ExtendPum/HE_ExtendPUM.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/mg_element.h
  ../Krylov/GMRES.h
)

# define executable
add_executable(Wave_Ray ${Wave_Ray_sources})

#link executable with LehrFEM++
target_link_libraries(Wave_Ray
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

#############################################################################

# set(E_Wave_Ray_sources
#   E_Wave_Ray.cpp
#   vcycle_function.cpp
#   ../ExtendPum_WaveRay/ExtendPUM_WaveRay.h
#   ../ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../LagrangeO1/HE_LagrangeO1.h
#   ../LagrangeO1/HE_LagrangeO1.cpp
#   ../ExtendPum/ExtendPUM_EdgeMat.h
#   ../ExtendPum/ExtendPUM_EdgeVector.h
#   ../ExtendPum/ExtendPUM_ElementMatrix.h
#   ../ExtendPum/ExtendPUM_ElemVector.h
#   ../ExtendPum/HE_ExtendPUM.h
#   ../ExtendPum/HE_ExtendPUM.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
# )

# # define executable
# add_executable(E_Wave_Ray ${E_Wave_Ray_sources})

# #link executable with LehrFEM++
# target_link_libraries(E_Wave_Ray
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