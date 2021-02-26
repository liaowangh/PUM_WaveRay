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

void mg_O1(HE_LagrangeO1& he_O1, int L, int nr_coarsemesh, FHandle_t u) {
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };

    auto dofh = he_O1.get_dofh(L);
    int N = dofh.NumDofs();

    Vec_t v = Vec_t::Random(N); // initial value
    Vec_t uh = he_O1.solve(L);

    int nu1 = 3, nu2 = 3;

    std::vector<double> L2_vk;
    std::vector<double> L2_ek;

    // std::cout << std::scientific << std::setprecision(1);
    for(int k = 0; k < 10; ++k) {
        std::cout << he_O1.L2_Err(L, v - uh, zero_fun) << " ";
        std::cout << he_O1.L2_Err(L, v, u) << std::endl;
        L2_vk.push_back(he_O1.L2_Err(L, v, zero_fun));
        L2_ek.push_back(he_O1.L2_Err(L, v - uh, zero_fun));
        he_O1.solve_multigrid(v, L, nr_coarsemesh, nu1, nu2, true);
    }
    std::cout << "||u-uh||_2 = " << he_O1.L2_Err(L, uh, u) << std::endl;
    std::cout << "||v_{k+1}||/||v_k||" << std::endl;
    for(int k = 0; k + 1 < L2_vk.size(); ++k) {
        std::cout << k << " " << L2_vk[k+1] / L2_vk[k] 
                       << " " << L2_ek[k+1] / L2_ek[k] << std::endl;
    }
} 


int main(){
    // mesh path
    // boost::filesystem::path here = __FILE__;
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/square.msh")).string();

    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/LagrangeO1/";
    size_type L = 5; // refinement steps
    double k = 32; // wave number
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;
    
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
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

        int num_coarserlayer = 3;
        mg_O1(he_O1, L, num_coarserlayer, u);
       
        // HE_LagrangeO1::Vec_t true_sol = he_O1.solve(L);
        // auto eq_pair = he_O1.build_equation(L);
        // auto A_crs(eq_pair.first.makeDense());
        // HE_LagrangeO1::Vec_t fem_sol = HE_LagrangeO1::Vec_t::Random(eq_pair.second.size());
        // Gaussian_Seidel(A_crs, eq_pair.second, fem_sol, true_sol, 1);
        // auto eigen_pair = power_GS(A_crs, 1);
    }
}