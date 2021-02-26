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

void ePUM_WaveRay(ExtendPUM_WaveRay& epum_waveray, int L, int nr_coarsemesh, FHandle_t u) {
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };

    auto dofh = epum_waveray.get_dofh(L);
    int N = dofh.NumDofs();

    Vec_t v = Vec_t::Random(N); // initial value
    Vec_t uh = epum_waveray.solve(L);

    int nu1 = 3, nu2 = 3;

    std::vector<double> L2_vk;
    std::vector<double> L2_ek;

    // std::cout << std::scientific << std::setprecision(1);
    for(int k = 0; k < 10; ++k) {
        std::cout << epum_waveray.L2_Err(L, v - uh, zero_fun) << " ";
        std::cout << epum_waveray.L2_Err(L, v, u) << std::endl;
        L2_vk.push_back(epum_waveray.L2_Err(L, v, zero_fun));
        L2_ek.push_back(epum_waveray.L2_Err(L, v - uh, zero_fun));
        epum_waveray.solve_multigrid(v, L, nr_coarsemesh, nu1, nu2, false);
    }
    std::cout << "||u-uh||_2 = " << epum_waveray.L2_Err(L, uh, u) << std::endl;
    std::cout << "||v_{k+1}||/||v_k||" << std::endl;
    for(int k = 0; k + 1 < L2_vk.size(); ++k) {
        std::cout << k << " " << L2_vk[k+1] / L2_vk[k] 
                       << " " << L2_ek[k+1] / L2_ek[k] << std::endl;
    }
} 

/*
 * For square_hole.msh, h = 2^{-L-1}, and we can choose k = 2^{L-1}*pi
 * For square.msh, h_L = =2^{-L}, and we choose k = 2^{L-1}, then h_L * k = 0.5
 */
int main(){
    std::string square_output = "../result_square/ExtendPUM_WaveRay/";
    std::string square_hole_output = "../result_squarehole/ExtendPUM_WaveRaay/";
    std::string square = "../meshes/square.msh";
    std::string square_hole = "../meshes/square_hole.msh";
    size_type L = 5; // refinement steps
 
    auto zero_fun = [](const coordinate_t& x)->Scalar { return 0.0; };

    double k = 16; // wave number
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);
 
    std::vector<int> num_planwaves(L+1);
    num_planwaves[L] = 2;
    for(int i = L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }
    for(int i = 0; i < solutions.size(); ++i) {
        if(i > 0){
            continue;
        }
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();
        ExtendPUM_WaveRay extend_waveray(L, k, square, g, u, false, num_planwaves, 50);
 
        int num_wavelayer = 1;
        ePUM_WaveRay(extend_waveray, L, num_wavelayer, u);
    }
}