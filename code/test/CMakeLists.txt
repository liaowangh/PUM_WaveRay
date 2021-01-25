set(numbering_sources
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  PUM_BasisNumbering.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
)

# define executable
add_executable(numbering ${numbering_sources})

#link executable with LehrFEM++
target_link_libraries(numbering
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

set(Laplace_sources
  ../LagrangeO1/HE_LagrangeO1.h
  ../LagrangeO1/HE_LagrangeO1.cpp
  Laplace.cpp
  ../utils/utils.h
  ../utils/utils.cpp
  ../utils/HE_solution.h
  ../utils/HE_solution.cpp
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
