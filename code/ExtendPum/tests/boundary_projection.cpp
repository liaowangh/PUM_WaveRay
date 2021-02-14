#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../HE_ExtendPUM.h"
#include "../../utils/HE_solution.h"
#include "../../utils/utils.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

int main() {
    // boost::filesystem::path here = __FILE__;
    // auto mesh_path = (here.parent_path().parent_path()).string(); 
    // std::cout << here.string() << std::endl;

    std::string square_hole = "../meshes/square_hole.msh";
    std::string output_folder = "../result_squarehole/ExtendPUM/";
    int L = 5; // refinement steps
    std::vector<int> number_waves{3, 5, 7, 9, 11, 13};
    
    double k = 20;

    // plan_wave sol(k, 0.8, 0.6);
    Spherical_wave sol(k, 2);
    auto u = sol.get_fun();
    auto grad_u = sol.get_gradient();
    auto g = sol.boundary_g();
    
    HE_ExtendPUM he_epum(L, k, square_hole, g, u, true, std::vector<int>(L+1, 3), 10);
    auto eq_pair = he_epum.build_equation(0);
    
}