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


void ray_cycle(Vec_t& v, ExtendPUM_WaveRay& epum_waveray, int L, int num_wavelayer, 
    double k, FHandle_t u, bool solve_coarest) {

    auto eq_pair = epum_waveray.build_equation(L);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(num_wavelayer + 1), prolongation_op(num_wavelayer);
    std::vector<int> stride(num_wavelayer + 1);
    std::vector<double> ms(num_wavelayer + 1);
    auto mesh_width = epum_waveray.mesh_width();
    Op[num_wavelayer] = A;
    stride[num_wavelayer] = 1;
    ms[num_wavelayer] = mesh_width[L];
    for(int i = num_wavelayer - 1; i >= 0; --i) {
        int idx = L + i - num_wavelayer;
        prolongation_op[i] = epum_waveray.prolongation(idx);
        // auto tmp = epum_waveray.build_equation(idx);
        // Op[i] = tmp.first.makeSparse();
        Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
        // stride[i] = epum_waveray.num_planwaves[idx] + 1;
        stride[i] = 1;
        ms[i] = mesh_width[idx];
    }
    /***********************************************************************/
    int nu1 = 3, nu2 = 3;
    v_cycle(v, eq_pair.second, Op, prolongation_op, stride, nu1, nu2, solve_coarest);
} 

void wave_ray(HE_LagrangeO1& he_O1, ExtendPUM_WaveRay& epum, int wave_start,
    int ray_start, int nr_raylayer, double k, FHandle_t u) {

    auto eq_pair = he_O1.build_equation(wave_start);
    SpMat_t A(eq_pair.first.makeSparse());
    int nr_coarsemesh = wave_start - ray_start - 1;
    std::vector<SpMat_t> Op(nr_coarsemesh + 1), prolongation_op(nr_coarsemesh);
    Op[nr_coarsemesh] = A;
    for(int i = nr_coarsemesh - 1; i >= 0; --i) {
        int idx = wave_start + i - nr_coarsemesh;
        prolongation_op[i] = he_O1.prolongation(idx);
        Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
    }

    Vec_t v = Vec_t::Random(A.rows());
    Vec_t uh = he_O1.solve(wave_start);  // finite element solution
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };

    std::vector<double> L2_vk, L2_ek;
    for(int j = 0; j < 10; ++j) {

        std::cout << he_O1.L2_Err(wave_start, v - uh, zero_fun) << " ";
        std::cout << he_O1.L2_Err(wave_start, v, u) << std::endl;
        L2_vk.push_back(he_O1.L2_Err(wave_start, v, zero_fun));
        L2_ek.push_back(he_O1.L2_Err(wave_start, v - uh, zero_fun));

        /************ first leg of wave cycle *************/
        int nu1 = 3, nu2 = 3;
        std::vector<int> op_size(nr_coarsemesh+1);
        for(int i = 0; i <= nr_coarsemesh; ++i) {
            op_size[i] = Op[i].rows();
        }
        std::vector<Vec_t> initial(nr_coarsemesh + 1), rhs_vec(nr_coarsemesh + 1);

        initial[nr_coarsemesh] = v;
        rhs_vec[nr_coarsemesh] = eq_pair.second;
        // initial guess on coarser mesh are all zero
        for(int i = 0; i < nr_coarsemesh; ++i) {
            initial[i] = Vec_t::Zero(op_size[i]);
        }
        for(int i = nr_coarsemesh; i > 0; --i) {
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], 1, nu1);
            rhs_vec[i-1] = prolongation_op[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
        }

        Eigen::SparseLU<SpMat_t> solver;
        solver.compute(Op[0]);
        initial[0] = solver.solve(rhs_vec[0]);
        // do the ray cycle
        // epum.solve_multigrid(initial[0], ray_start + 1, nr_raylayer, nu1, nu2, false);

        /************ second leg of wave cycle *************/
        for(int i = 1; i <= nr_coarsemesh; ++i) {
            initial[i] += prolongation_op[i-1] * initial[i-1];
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], 1, nu2);
        }
        v = initial[nr_coarsemesh];
        /* finish one iteration of wave-ray*/
    }
    std::cout << "||u-uh||_2 = " << he_O1.L2_Err(wave_start, uh, u) << std::endl;
    std::cout << "||v_{k+1}||/||v_k||" << std::endl;
    for(int j = 0; j + 1 < L2_vk.size(); ++j) {
        std::cout << j << " " << L2_vk[j+1] / L2_vk[j] 
                       << " " << L2_ek[j+1] / L2_ek[j] << std::endl;
    }
}

int main(){
    std::string square_output = "../result_square/ExtendPUM_WaveRay/";
    std::string square_hole_output = "../result_squarehole/ExtendPUM_WaveRaay/";
    std::string square = "../meshes/square.msh";
    std::string square_hole = "../meshes/square_hole.msh";
    std::string square_hole2 = "../meshes/square_hole2.msh";

    size_type wave_L = 5, ray_L = 3; // refinement steps
    double k = 2.0; // wave number
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

    int nr_raylayer = 1;
    wave_ray(he_O1, epum, wave_L, ray_L - 1, nr_raylayer, k, u);
}