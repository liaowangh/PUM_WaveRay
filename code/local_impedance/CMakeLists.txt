set(test_impedance_sources
  test_impedance.cpp
  local_impedance_solver.h
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../ExtendPum/ExtendPUM_ElementMatrix.h
  ../ExtendPum/ExtendPUM_EdgeMat.h
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(test_impedance ${test_impedance_sources})

#link executable with LehrFEM++
target_link_libraries(test_impedance
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

set(local_impedance_smoothing_sources
  local_impedance_smoothing.cpp
  local_impedance_solver.h
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../ExtendPum/ExtendPUM_ElementMatrix.h
  ../ExtendPum/ExtendPUM_EdgeMat.h
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

#############################################################################

set(waveray_impedance_sources
  WaveRay_Impedance.cpp
  local_impedance_solver.h
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  ../Pum_WaveRay/HE_FEM.h
  ../Pum_WaveRay/HE_FEM.cpp
  ../ExtendPum_WaveRay/ExtendPUM_WaveRay.h
  ../ExtendPum_WaveRay/ExtendPUM_WaveRay.cpp
  ../ExtendPum/HE_ExtendPUM.h
  ../ExtendPum/HE_ExtendPUM.cpp
  ../ExtendPum/ExtendPUM_ElementMatrix.h
  ../ExtendPum/ExtendPUM_EdgeMat.h
  ../ExtendPum/ExtendPUM_EdgeVector.h
  ../ExtendPum/ExtendPUM_ElemVector.h
  ../Krylov/GMRES.h
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(waveray_impedance ${waveray_impedance_sources})

#link executable with LehrFEM++
target_link_libraries(waveray_impedance
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