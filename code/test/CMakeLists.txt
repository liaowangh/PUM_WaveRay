# set(numbering_sources
#   ../LagrangeO1/HE_LagrangeO1.h
#   ../LagrangeO1/HE_LagrangeO1.cpp
#   PUM_BasisNumbering.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(numbering ${numbering_sources})

# #link executable with LehrFEM++
# target_link_libraries(numbering
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

####################################################################################

# set(prolongation_sources
#   test_prolongation.cpp
#   ../LagrangeO1/HE_LagrangeO1.h
#   ../LagrangeO1/HE_LagrangeO1.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../Pum_WaveRay/PUM_WaveRay.h
#   ../Pum_WaveRay/PUM_WaveRay.cpp
#   ../Pum/PUM_EdgeMat.h
#   ../Pum/PUM_EdgeMat.cpp
#   ../Pum/PUM_EdgeVector.h
#   ../Pum/PUM_EdgeVector.cpp
#   ../Pum/PUM_ElementMatrix.h
#   ../Pum/PUM_ElementMatrix.cpp
#   ../Pum/PUM_ElemVector.h
#   ../Pum/PUM_ElemVector.cpp
#   ../Pum/HE_PUM.h
#   ../Pum/HE_PUM.cpp
#   ../ExtendPum/ExtendPUM_EdgeMat.h
#   ../ExtendPum/ExtendPUM_EdgeMat.cpp
#   ../ExtendPum/ExtendPUM_EdgeVector.h
#   ../ExtendPum/ExtendPUM_EdgeVector.cpp
#   ../ExtendPum/ExtendPUM_ElementMatrix.h
#   ../ExtendPum/ExtendPUM_ElementMatrix.cpp
#   ../ExtendPum/ExtendPUM_ElemVector.h
#   ../ExtendPum/ExtendPUM_ElemVector.cpp
#   ../ExtendPum/HE_ExtendPUM.h
#   ../ExtendPum/HE_ExtendPUM.cpp
#   ../ExtendPum_WaveRay/ExtendPUM_WaveRay.h
#   ../ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(prolongation ${prolongation_sources})

# #link executable with LehrFEM++
# target_link_libraries(prolongation
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

####################################################################################

# set(quadrature_sources
#   test_quadrature.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
# )

# # define executable
# add_executable(quadrature ${quadrature_sources})

# #link executable with LehrFEM++
# target_link_libraries(quadrature
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

####################################################################################

# set(relaxation_sources
#   test_relaxation.cpp
#   ../LagrangeO1/HE_LagrangeO1.h
#   ../LagrangeO1/HE_LagrangeO1.cpp
#   ../Pum_WaveRay/HE_FEM.h
#   ../Pum_WaveRay/HE_FEM.cpp
#   ../ExtendPum/ExtendPUM_EdgeMat.h
#   ../ExtendPum/ExtendPUM_EdgeMat.cpp
#   ../ExtendPum/ExtendPUM_EdgeVector.h
#   ../ExtendPum/ExtendPUM_EdgeVector.cpp
#   ../ExtendPum/ExtendPUM_ElementMatrix.h
#   ../ExtendPum/ExtendPUM_ElementMatrix.cpp
#   ../ExtendPum/ExtendPUM_ElemVector.h
#   ../ExtendPum/ExtendPUM_ElemVector.cpp
#   ../ExtendPum/HE_ExtendPUM.h
#   ../ExtendPum/HE_ExtendPUM.cpp
#   ../ExtendPum_WaveRay/ExtendPUM_WaveRay.h
#   ../ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
# )

# # define executable
# add_executable(relaxation ${relaxation_sources})

# #link executable with LehrFEM++
# target_link_libraries(relaxation
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

####################################################################################

set(Laplace_sources
  test_Laplace.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
)

# define executable
add_executable(Laplace ${Laplace_sources})

#link executable with LehrFEM++
target_link_libraries(Laplace
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