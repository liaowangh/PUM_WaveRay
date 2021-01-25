#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/filesystem.hpp>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../LagrangeO1/HE_LagrangeO1.h"
#include "../utils/HE_solution.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

int main(){
    // mesh path
    boost::filesystem::path here = __FILE__;
    auto mesh_path = (here.parent_path().parent_path() / ("meshes/coarest_mesh.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/no_hole.msh")).string(); 
    size_type L = 5; // refinement steps
    double k = 0; // Laplace equation
   
    std::string sol_name = "Harmonic_function";
    Harmonic_fun hf;
    auto u = hf.get_fun();
    auto g = hf.boundary_g();
    HE_LagrangeO1 he_O1(L, k, mesh_path, g, u, true);
    // solve_directly(he_O1, sol_name, L, u);
    return 0;
}