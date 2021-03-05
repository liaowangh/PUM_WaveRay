# set(test_pum_waveray_sources
#   HE_FEM.h
#   HE_FEM.cpp
#   PUM_WaveRay.h
#   PUM_WaveRay.cpp
#   tests/test_waveray.cpp
#   ../LagrangeO1/HE_LagrangeO1.h
#   ../LagrangeO1/HE_LagrangeO1.cpp
#   ../Pum/PUM_EdgeMat.h
#   ../Pum/PUM_EdgeVector.h
#   ../Pum/PUM_ElementMatrix.h
#   ../Pum/PUM_ElemVector.h
#   ../Pum/HE_PUM.h
#   ../Pum/HE_PUM.cpp
#   ../utils/HE_solution.h
#   ../utils/HE_solution.cpp
#   ../utils/utils.h
#   ../utils/utils.cpp
# )

# # define executable
# add_executable(test_pum_waveray ${test_pum_waveray_sources})

# #link executable with LehrFEM++
# target_link_libraries(test_pum_waveray
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

##################################################################

set(relaxation_sources
  HE_FEM.h
  HE_FEM.cpp
  PUM_WaveRay.h
  PUM_WaveRay.cpp
  tests/waveray_relaxation.cpp
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../Pum/PUM_EdgeMat.h
  ../Pum/PUM_EdgeVector.h
  ../Pum/PUM_ElementMatrix.h
  ../Pum/PUM_ElemVector.h
  ../Pum/HE_PUM.h
  ../Pum/HE_PUM.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
  ../utils/utils.h
  ../utils/utils.cpp
)

# define executable
add_executable(relaxation ${relaxation_sources})

#link executable with LehrFEM++
target_link_libraries(relaxation
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