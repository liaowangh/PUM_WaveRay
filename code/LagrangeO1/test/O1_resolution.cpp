#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../HE_LagrangeO1.h"
#include "../../utils/HE_solution.h"
#include "../../utils/utils.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

/*
 * Convergence studies
 *  different wave numbers: k = 6, 20, 60
 *  on sequences of meshes (refinement steps: 5)
 *  different number of plan waves (0, 3, 5, 7, 9, 11, 13)
 *  compute L2 error norm and H1 error semi-norm
 */
int main() {
    std::string square_hole = "../meshes/square_hole.msh";
    std::string square = "../meshes/square.msh";
    std::string square_hole_output = "../result_squarehole/LagrangeO1/";
    std::string square_output = "../result_square/LagrangeO1/";
    int L = 5; // refinement steps
    
    std::vector<double> wave_number{6,20,60};
    //double k = 6; // wave number in Helmholtz equation

    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;
    
    std::vector<std::string> sol_name{"plan_wave", "fundamental_sol", "spherical_wave"};
   
    for(auto k: wave_number) {
        std::vector<std::shared_ptr<HE_sol>> solutions(3);
        solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
        solutions[1] = std::make_shared<fundamental_sol>(k, c);
        solutions[2] = std::make_shared<Spherical_wave>(k, 2);
        for(int i = 0; i < solutions.size(); ++i) {
            if(i > 0) { continue; }
            auto u = solutions[i]->get_fun();
            auto g = solutions[i]->boundary_g();
            auto grad_u = solutions[i]->get_gradient();
            std::string str = "k" + std::to_string(int(k)) + "_" + sol_name[i];
            HE_LagrangeO1 he_O1(L, k, square, g, u, false);
            test_solve(he_O1, str, square_output, L, u, grad_u);
            std::cout << std::endl;
        }
    }
}

