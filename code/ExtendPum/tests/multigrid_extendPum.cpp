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
    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/ExtendPUM/";
    int L = 3; // refinement steps
    double k = 2.0; // wave number
    
    std::vector<int> num_planwaves(L+1);
    num_planwaves[L] = 4;
    for(int i = L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }

    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);
    
    std::vector<std::string> sol_name{"plan_wave", "fundamental_sol", "spherical_wave"};

    for(int i = 0; i < solutions.size(); ++i) {
        if(i > 0) { continue; }
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();

        HE_ExtendPUM he_epum(L, k, square, g, u, false, num_planwaves, 20);

        auto eq_pair = he_epum.build_equation(L);
        SpMat_t A = eq_pair.first.makeSparse();
        int stride = num_planwaves[L]+1;
        power_GS(A, stride);

        // int num_wavelayer = 2;
        // he_epum.power_multigird(L, num_wavelayer, 10, 10);
        // Vec_t fem_sol = he_epum.solve_multigrid(L, num_wavelayer, 10, 10);
        // Vec_t true_sol = he_epum.solve(L);
        // std::cout << he_epum.mesh_width()[L] << " "
        //           << he_epum.L2_Err(L, fem_sol, u) << " " 
        //           << he_epum.H1_Err(L, fem_sol, u, grad_u) << " " 
        //           << he_epum.L2_Err(L, true_sol, u) << std::endl;
    }
    
}

