#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "local_impedance_solver.h"
#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../ExtendPum_WaveRay/ExtendPUM_WaveRay.h"

// everyting needs change

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

void E_wave_ray(HE_LagrangeO1& he_O1, ExtendPUM_WaveRay& epum, int wave_start, int wave_coarselayers,
    int ray_start, int ray_coarselayers, double k, FHandle_t u) {

    auto eq_pair = he_O1.build_equation(wave_start);
    SpMat_t wave_A(eq_pair.first.makeSparse());
    std::vector<SpMat_t> wave_op(wave_coarselayers + 1), wave_I(wave_coarselayers);
    wave_op[wave_coarselayers] = wave_A;
    std::vector<double> wave_ms(wave_coarselayers + 1);
    auto mesh_width1 = he_O1.mesh_width();
    for(int i = wave_coarselayers - 1; i >= 0; --i) {
        int idx = wave_start + i - wave_coarselayers;
        wave_I[i] = he_O1.prolongation(idx);
        wave_op[i] = wave_I[i].transpose() * wave_op[i+1] * wave_I[i];
        wave_ms[i] = mesh_width1[idx];
    }

    std::vector<SpMat_t> ray_op(ray_coarselayers + 1), ray_I(ray_coarselayers);
    ray_op[ray_coarselayers] = epum.build_equation(ray_start).first.makeSparse();
    // ray_op[ray_coarselayers] = wave_op[ray_start - wave_start + wave_coarselayers];
    std::vector<double> ray_ms(ray_coarselayers + 1);
    std::vector<int> stride(ray_coarselayers+1);
    stride[ray_coarselayers] = 1;
    auto mesh_width2 = epum.mesh_width();
    for(int i = ray_coarselayers - 1; i >= 0; --i) {
        int idx = ray_start + i - ray_coarselayers;
        ray_I[i] = epum.prolongation(idx);
        ray_op[i] = ray_I[i].transpose() * ray_op[i+1] * ray_I[i];
        ray_ms[i] = mesh_width2[idx];
        stride[i] =  epum.num_planwaves[idx] + 1
    }

    std::vector<double> L2_vk, L2_ek;
    Vec_t v = Vec_t::Random(wave_A.rows());
    Vec_t uh = he_O1.solve(wave_start);  // finite element solution
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };
    
    std::cout << std::setw(11) << "||v-uh||_2" << std::setw(11) << "||v-u||_2" << std::endl;
    for(int j = 0; j < 10; ++j) {
        std::cout << std::setw(11) << he_O1.L2_Err(wave_start, v - uh, zero_fun) << " ";
        std::cout << std::setw(11) << he_O1.L2_Err(wave_start, v, u) << std::endl;
        L2_vk.push_back(he_O1.L2_Err(wave_start, v, zero_fun));
        L2_ek.push_back(he_O1.L2_Err(wave_start, v - uh, zero_fun));
        pum_wave_cycle(v, eq_pair.second, wave_op, ray_op, wave_I, ray_I, k, 
            stride, wave_ms, ray_ms, wave_start, ray_start);
    }

    std::cout << "||u-uh||_2 = " << he_O1.L2_Err(wave_start, uh, u) << std::endl;
    std::cout << "||v_{k+1}||/||v_k||" << std::endl;
    std::cout << "j " 
        << std::setw(20) << "||v_{j+1}||/||v_j||" 
        << std::setw(20) << "||e_{j+1}||/||e_j||" << std::endl;
    for(int j = 0; j + 1 < L2_vk.size(); ++j) {
        std::cout << j << " " << std::setw(20) << L2_vk[j+1] / L2_vk[j] 
                       << " " << std::setw(20) << L2_ek[j+1] / L2_ek[j] << std::endl;
    }
}

int main(){
    std::string square = "../meshes/square.msh";
    std::string square_hole2 = "../meshes/square_hole2.msh";
    std::string triangle_hole = "../meshes/triangle_hole.msh";

    size_type wave_L = 5, ray_L = 3; // refinement steps
    double k = 20.0; // wave number
    std::vector<int> num_planwaves(ray_L+1);
    num_planwaves[ray_L] = 2;
    for(int i = ray_L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }

    auto zero_fun = [](const coordinate_t& x)->Scalar { return 0.0; };
    plan_wave sol(k, 0.8, 0.6);
    auto u = sol.get_fun();
    auto grad_u = sol.get_gradient();
    auto g = sol.boundary_g();

    HE_LagrangeO1 he_O1(wave_L, k, square, g, u, false, 50);
    ExtendPUM_WaveRay epum(ray_L, k, square, g, u, false, num_planwaves, 50);
    // HE_LagrangeO1 he_O1(wave_L, k, square_hole2, g, u, true, 50);
    // ExtendPUM_WaveRay epum(ray_L, k, square_hole2, g, u, true, num_planwaves, 50);
    // HE_LagrangeO1 he_O1(wave_L, k, triangle_hole, g, u, false, 50);
    // ExtendPUM_WaveRay epum(ray_L, k, triangle_hole, g, u, false, num_planwaves, 50);

    int wave_coarselayers = 5, ray_coarselayers = 1;
    E_wave_ray(he_O1, epum, wave_L, wave_coarselayers, ray_L, ray_coarselayers, k, u);
}
