set(O1_shift_sources
  O1_shift.cpp
  domain_mass_mat.h
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../Pum_WaveRay/PUM_WaveRay.h
  ../Pum_WaveRay/PUM_WaveRay.cpp
  ../Pum/PUM_EdgeMat.h
  ../Pum/PUM_EdgeVector.h
  ../Pum/PUM_ElementMatrix.h
  ../Pum/PUM_ElemVector.h
  ../Pum/HE_PUM.h
  ../Pum/HE_PUM.cpp
  ../ExtendPum/ExtendPUM_EdgeMat.h
  ../ExtendPum/ExtendPUM_EdgeVector.h
  ../ExtendPum/ExtendPUM_ElementMatrix.h
  ../ExtendPum/ExtendPUM_ElemVector.h
  ../ExtendPum/HE_ExtendPUM.h
  ../ExtendPum/HE_ExtendPUM.cpp
  ../ExtendPum_WaveRay/ExtendPUM_WaveRay.h
  ../ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(O1_shift ${O1_shift_sources})

#link executable with LehrFEM++
target_link_libraries(O1_shift
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