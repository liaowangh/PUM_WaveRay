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

#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "HE_LagrangeO1.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

int main(){
    // mesh path
    boost::filesystem::path here = __FILE__;
    auto mesh_path = (here.parent_path().parent_path() / ("meshes/coarest_mesh.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/no_hole.msh")).string();
    size_type L = 5; // refinement steps
    double k = 2; // wave number
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;
    
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.6, 0.8);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);
 
    std::vector<std::string> sol_name{"plan_wave", "fundamental_sol", "spherical_wave"};
    for(int i = 0; i < 3; ++i) {
        auto u = solutions[i]->get_fun();
        auto g = solutions[i]->boundary_g();
        HE_LagrangeO1 he_O1(L, k, mesh_path, g, u, true);
        solve_directly(he_O1, sol_name[i], L, u);
        std::cout << std::endl;
    }
}