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

std::pair<Vec_t, Scalar> 
power_multigird(HE_FEM& he_fem, size_type start_layer, int num_coarserlayer, 
    std::vector<int>& stride, int mu1, int mu2, bool verbose) {

    LF_ASSERT_MSG((num_coarserlayer <= start_layer), 
        "please use a smaller number of wave layers");

    auto eq_pair = he_fem.build_equation(start_layer);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
    Op[num_coarserlayer] = A;
    for(int i = num_coarserlayer - 1; i >= 0; --i) {
        int idx = start_layer + i - num_coarserlayer;
        prolongation_op[i] = he_fem.prolongation(idx);
        Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
    }

    /***************************************/
    int N = A.rows();
    Vec_t u = Vec_t::Random(N);
    u.normalize();
    Vec_t old_u;
    Vec_t zero_vec = Vec_t::Zero(N);
    Scalar lambda;
    int cnt = 0;
    
    if(verbose) {
        std::cout << std::left << std::setw(10) << "Iteration" 
            << std::setw(20) << "residual_norm" << std::endl;
    }
    
    while(true) {
        cnt++;
        old_u = u;
        v_cycle(u, zero_vec, Op, prolongation_op, stride, mu1, mu2);
        
        lambda = old_u.dot(u);  // domainant eigenvalue
        auto r = u - lambda * old_u;
        double r_norm = r.norm();
        u.normalize();
    
        if(verbose && cnt % 5 == 0) {
            std::cout << std::left << std::setw(10) << cnt 
                << std::setw(20) << r_norm
                << std::setw(20) << (u - old_u).norm()
                << std::endl;
        }
        
        if(r_norm < 0.01) {
            if(verbose) {
                std::cout << "Power iteration converges after " << cnt << " iterations" << std::endl;
            }
            break;
        }
        if(cnt > 30) {
            if(verbose) {
                std::cout << "Power iteration for multigrid doesn't converge." << std::endl;
            }
            break;
        }
    }
    if(verbose) {
        std::cout << "Number of iterations: " << cnt << std::endl;
        std::cout << "Domainant eigenvalue by power iteration: " << lambda << std::endl;
    }
    return std::make_pair(u, lambda);
}

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