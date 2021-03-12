#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../../utils/HE_solution.h"
#include "../../utils/utils.h"
#include "../ExtendPUM_WaveRay.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

int main(){
    std::string square_output = "../result_square/ExtendPUM_WaveRay/";
    std::string square_hole_output = "../result_squarehole/ExtendPUM_WaveRaay/";
    std::string square = "../meshes/square.msh";
    std::string square_hole = "../meshes/square_hole.msh";
    std::string triangle_hole = "../meshes/triangle_hole.msh";
    std::string square_hole2 = "../meshes/square_hole2.msh";
    size_type L = 4; // refinement steps
 
    std::vector<double> wave_number;
    for(int i = 2; i <= 144; i += 2) {
        wave_number.push_back(i);
    }
    // for(int i = 90; i <= 160; i += 2) {
    //     wave_number.push_back(i);
    // }

    std::vector<int> num_planwaves(L+1);
    num_planwaves[L] = 2;
    for(int i = L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }

    int num_wavelayer = 2;
    std::vector<int> stride(num_wavelayer + 1);
    stride[num_wavelayer] = 1;
    for(int i = 0; i < num_wavelayer - 1; ++i) {
        // stride[i] = num_planwaves[L+i-num_wavelayer] + 1;
        stride[i] = 1;
    }

    std::cout << std::left;
    std::cout << std::setw(10) << "k * h" << std::setw(10) << "convergence factor" << std::endl;
    for(auto i : wave_number) {
        double k = i;
        plan_wave sol(k, 0.8, 0.6);
        auto u = sol.get_fun();
        auto grad_u = sol.get_gradient();
        auto g = sol.boundary_g();

        // ExtendPUM_WaveRay extend_waveray(L, k, square, g, u, false, num_planwaves, 50);
        // ExtendPUM_WaveRay extend_waveray(L, k, square_hole2, g, u, true, num_planwaves, 50);
        // ExtendPUM_WaveRay extend_waveray(L, k, square_hole, g, u, true, num_planwaves, 50);
        ExtendPUM_WaveRay extend_waveray(L, k, triangle_hole, g, u, true, num_planwaves, 50);

        // if(k != 100) {
        //     continue;
        // } else {
        //     auto eq_pair = extend_waveray.build_equation(L);
        //     Mat_t A = eq_pair.first.makeDense();
        //     for(int j = 0; j < A.rows(); ++j) {
        //         std::cout << j << " " << std::abs(A(j,j)) << std::endl;
        //     }
        //     break;
        // }
       
        auto ei_pair = power_multigird(extend_waveray, L, num_wavelayer, stride, 3, 3, false);
        double hL = extend_waveray.mesh_width()[L];
        std::cout << std::setw(10) << k * hL << std::setw(10) << std::abs(ei_pair.second) << std::endl;

        // Vec_t uh = extend_waveray.solve(L); // finite element solution
        // Vec_t vh = Vec_t::Zero(uh.size());
        
        // extend_waveray.solve_multigrid(vh, L, 1, 3, 3, true);
        // std::cout << std::setw(10) << k * hL << std::setw(10) << (uh - vh).norm() << std::endl;
    } 
}