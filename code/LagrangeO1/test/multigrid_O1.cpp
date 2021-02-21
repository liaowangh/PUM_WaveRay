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

#include "../../utils/HE_solution.h"
#include "../../utils/utils.h"
#include "../HE_LagrangeO1.h"

using namespace std::complex_literals;

int main(){
    // mesh path
    // boost::filesystem::path here = __FILE__;
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/square.msh")).string();

    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/LagrangeO1/";
    size_type L = 3; // refinement steps
    double k = 0.5; // wave number
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;
    
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.6, 0.8);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);
 
    std::vector<std::string> sol_name{"plan_wave", "fundamental_sol", "spherical_wave"};
    for(int i = 0; i < solutions.size(); ++i) {
        if(i > 0) {
            continue;
        }
        auto u = solutions[i]->get_fun();
        auto g = solutions[i]->boundary_g();
        auto grad_u = solutions[i]->get_gradient();
        HE_LagrangeO1 he_O1(L, k, square, g, u, false);

        int num_coarserlayer = 2;
        he_O1.power_multigird(L, num_coarserlayer, 10, 10);
        Vec_t fem_sol = he_O1.solve_multigrid(L, num_coarserlayer, 10, 10);
        std::cout << he_O1.mesh_width()[L] << " "
                  << he_O1.L2_Err(L, fem_sol, u) << " " 
                  << he_O1.H1_Err(L, fem_sol, u, grad_u) << std::endl;
       
        // HE_LagrangeO1::Vec_t true_sol = he_O1.solve(L);
        // auto eq_pair = he_O1.build_equation(L);
        // auto A_crs(eq_pair.first.makeDense());
        // HE_LagrangeO1::Vec_t fem_sol = HE_LagrangeO1::Vec_t::Random(eq_pair.second.size());
        // Gaussian_Seidel(A_crs, eq_pair.second, fem_sol, true_sol, 1);
        // auto eigen_pair = power_GS(A_crs, 1);
    }
}