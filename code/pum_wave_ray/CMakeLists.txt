set(test_pum_waveray_sources
  HE_FEM.h
  HE_FEM.cpp
  PUM_WaveRay.h
  PUM_WaveRay.cpp
  test_waveray.cpp
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../planwave_pum/PUM_EdgeMat.h
  ../planwave_pum/PUM_EdgeMat.cpp
  ../planwave_pum/PUM_EdgeVector.h
  ../planwave_pum/PUM_EdgeVector.cpp
  ../planwave_pum/PUM_ElementMatrix.h
  ../planwave_pum/PUM_ElementMatrix.cpp
  ../planwave_pum/PUM_ElemVector.h
  ../planwave_pum/PUM_ElemVector.cpp
  ../planwave_pum/HE_PUM.h
  ../planwave_pum/HE_PUM.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
  ../utils/utils.h
  ../utils/utils.cpp
)

# define executable
add_executable(test_pum_waveray ${test_pum_waveray_sources})

#link executable with LehrFEM++
target_link_libraries(test_pum_waveray
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