set(singlePUM_sources
  ../pum_wave_ray/HE_FEM.h
  PUM_EdgeMat.h
  PUM_EdgeMat.cpp
  PUM_EdgeVector.h
  PUM_EdgeVector.cpp
  PUM_ElementMatrix.h
  PUM_ElementMatrix.cpp
  PUM_ElemVector.h
  PUM_ElemVector.cpp
  HE_PUM.h
  HE_PUM.cpp
  singlemesh_pum.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(singlePUM ${singlePUM_sources})

#link executable with LehrFEM++
target_link_libraries(singlePUM
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