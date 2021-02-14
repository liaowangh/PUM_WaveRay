set(test_extendpum_waveray_sources
  ../pum_wave_ray/HE_FEM.h
  ../pum_wave_ray/HE_FEM.cpp
  ExtendPUM_WaveRay.h
  ExtendPUM_WaveRay.cpp
  test_ExtendPUM_WaveRay.cpp
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../ExtendPum/ExtendPUM_EdgeMat.h
  ../ExtendPum/ExtendPUM_EdgeMat.cpp
  ../ExtendPum/ExtendPUM_EdgeVector.h
  ../ExtendPum/ExtendPUM_EdgeVector.cpp
  ../ExtendPum/ExtendPUM_ElementMatrix.h
  ../ExtendPum/ExtendPUM_ElementMatrix.cpp
  ../ExtendPum/ExtendPUM_ElemVector.h
  ../ExtendPum/ExtendPUM_ElemVector.cpp
  ../ExtendPum/HE_ExtendPUM.h
  ../ExtendPum/HE_ExtendPUM.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
  ../utils/utils.h
  ../utils/utils.cpp
)

# define executable
add_executable(test_extendpum_waveray ${test_extendpum_waveray_sources})

#link executable with LehrFEM++
target_link_libraries(test_extendpum_waveray
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