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

void ray_cycle(Vec_t& v, Vec_t& phi, std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, 
    std::vector<int>& stride, std::vector<double>& ms, double k, bool solve_coarest) {

    int nu1 = 3, nu2 = 3;
    int nr_raylayer = I.size();
    std::vector<int> op_size(nr_raylayer+1);
    for(int i = 0; i <= nr_raylayer; ++i) {
        op_size[i] = Op[i].rows();
    }

    std::vector<Vec_t> initial(nr_raylayer + 1), rhs_vec(nr_raylayer + 1);
    initial[nr_raylayer] = v;
    rhs_vec[nr_raylayer] = phi;
    // initial guess on coarser mesh are all zero
    for(int i = 0; i < nr_raylayer; ++i) {
        initial[i] = Vec_t::Zero(op_size[i]);
    }
    for(int i = nr_raylayer; i > 0; --i) {
        if(i == nr_raylayer + 1 || (k * ms[i] >= 1.5 && k * ms[i] <= 8.0)) {
            Kaczmarz(Op[i], rhs_vec[i], initial[i], 5 * nu1);
        } else {
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], nu1);
        }
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }

    if(solve_coarest) {
        Eigen::SparseLU<SpMat_t> solver;
        solver.compute(Op[0]);
        initial[0] = solver.solve(rhs_vec[0]);
    } else {
        if(k * ms[0] < 2.0 || k * ms[0] > 6.0){
            Gaussian_Seidel(Op[0], rhs_vec[0], initial[0], stride[0], nu1 + nu2);
            // block_GS(Op[i], rhs_vec[i], initial[i], stride[i], nu2);
        } else {
            Kaczmarz(Op[0], rhs_vec[0], initial[0], 5 * nu2);
        }
    }

    for(int i = 1; i <= nr_raylayer; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        if(k * ms[i] < 2.0 || k * ms[i] > 6.0){
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], nu2);
            // block_GS(Op[i], rhs_vec[i], initial[i], stride[i], nu2);
        } else {
            Kaczmarz(Op[i], rhs_vec[i], initial[i], 5 * nu2);
        }
    }
    v = initial[nr_raylayer];
} 

void wave_ray(HE_LagrangeO1& he_O1, ExtendPUM_WaveRay& epum, int wave_start, int wave_coarselayers,
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
        // wave_op[i] = wave_I[i].transpose() * wave_op[i+1] * wave_I[i];
        auto tmp = he_O1.build_equation(idx);
        wave_op[i] = tmp.first.makeSparse();
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
        // auto tmp = epum.build_equation(idx);
        // ray_op[i] = tmp.first.makeSparse();
        ray_ms[i] = mesh_width2[idx];
    }

    /***** start wave-ray *****/
    Vec_t v = Vec_t::Random(wave_A.rows());
    Vec_t uh = he_O1.solve(wave_start);  // finite element solution
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };

    std::vector<double> L2_vk, L2_ek;
    int nu1 = 1, nu2 = 1;
    std::vector<int> op_size(wave_coarselayers+1);
    for(int i = 0; i <= wave_coarselayers; ++i) {
        op_size[i] = wave_op[i].rows();
    }
    std::vector<Vec_t> initial(wave_coarselayers + 1), rhs_vec(wave_coarselayers + 1);
    std::cout << std::setw(11) << "||v-uh||_2" << std::setw(11) << "||v-u||_2" << std::endl;
    for(int j = 0; j < 10; ++j) {
        std::cout << std::setw(11) << he_O1.L2_Err(wave_start, v - uh, zero_fun) << " ";
        std::cout << std::setw(11) << he_O1.L2_Err(wave_start, v, u) << std::endl;
        L2_vk.push_back(he_O1.L2_Err(wave_start, v, zero_fun));
        L2_ek.push_back(he_O1.L2_Err(wave_start, v - uh, zero_fun));

        /************ first leg of wave cycle *************/
        initial[wave_coarselayers] = v;
        rhs_vec[wave_coarselayers] = eq_pair.second;
        // initial guess on coarser mesh are all zero
        for(int i = 0; i < wave_coarselayers; ++i) {
            initial[i] = Vec_t::Zero(op_size[i]);
        }
        for(int i = wave_coarselayers; i > 0; --i) {
            if(k * wave_ms[i] < 1.5 || k * wave_ms[i] > 8.0){
                Gaussian_Seidel(wave_op[i], rhs_vec[i], initial[i], 1, nu1);
            } else {
                // Kaczmarz(wave_op[i], rhs_vec[i], initial[i], 5 * nu1);
            }
            rhs_vec[i-1] = wave_I[i-1].transpose() * (rhs_vec[i] - wave_op[i] * initial[i]);
        }

        Eigen::SparseLU<SpMat_t> solver;
        solver.compute(wave_op[0]);
        initial[0] = solver.solve(rhs_vec[0]);

        /************ second leg of wave cycle *************/
        for(int i = 1; i <= wave_coarselayers; ++i) {
            initial[i] += wave_I[i-1] * initial[i-1];
            if(wave_start + i - wave_coarselayers == ray_start){
                ray_cycle(initial[i], rhs_vec[i], ray_op, ray_I, stride, ray_ms, k, true);
            } else if(k * wave_ms[i] < 1.5 || k * wave_ms[i] > 8.0){
                Gaussian_Seidel(wave_op[i], rhs_vec[i], initial[i], 1, nu2);
                // block_GS(wave_op[i], rhs_vec[i], initial[i], stride[i], nu2);
            } else {
                // Kaczmarz(wave_op[i], rhs_vec[i], initial[i], 5 * nu2);
            }
        }
        v = initial[wave_coarselayers];
        /* finish one iteration of wave-ray */
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
    std::string square_output = "../result_square/ExtendPUM_WaveRay/";
    std::string square_hole_output = "../result_squarehole/ExtendPUM_WaveRaay/";
    std::string square = "../meshes/square.msh";
    std::string square_hole = "../meshes/square_hole.msh";
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

    // HE_LagrangeO1 he_O1(wave_L, k, square, g, u, false, 50);
    // ExtendPUM_WaveRay epum(ray_L, k, square, g, u, false, num_planwaves, 50);
    HE_LagrangeO1 he_O1(wave_L, k, square_hole2, g, u, true, 50);
    ExtendPUM_WaveRay epum(ray_L, k, square_hole2, g, u, true, num_planwaves, 50);
    // HE_LagrangeO1 he_O1(wave_L, k, triangle_hole, g, u, false, 50);
    // ExtendPUM_WaveRay epum(ray_L, k, triangle_hole, g, u, false, num_planwaves, 50);

    int wave_coarselayers = 5, ray_coarselayers = 1;
    wave_ray(he_O1, epum, wave_L, wave_coarselayers, ray_L, ray_coarselayers, k, u);
}