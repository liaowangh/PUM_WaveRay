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
#include "../PUM_WaveRay.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

std::pair<Vec_t, Scalar> power_waveray(SpMat_t& A, int stride) {
    Mat_t dense_A = Mat_t(A);
    Mat_t A_L = Mat_t(dense_A.triangularView<Eigen::Lower>());
    Mat_t A_U = A_L - dense_A;
    Mat_t GS_op = A_L.colPivHouseholderQr().solve(A_U);
    Vec_t eivals = GS_op.eigenvalues();
    Scalar domainant_eival = eivals(0);
    for(int i = 1; i < eivals.size(); ++i) {
        if(std::abs(eivals(i)) > std::abs(domainant_eival)) {
            domainant_eival = eivals(i);
        }
    }
    // std::cout << eivals << std::endl;
    std::cout << "Domainant eigenvalue: " << domainant_eival << std::endl;
    std::cout << "Absolute value: " << std::abs(domainant_eival) << std::endl;

    /************************************************************************/
    double tol = 0.001;
    int N = A.rows();
    Vec_t u = Vec_t::Random(N);

    u.normalize();    
    Scalar lambda;
    int cnt = 0;

    std::cout << std::left << std::setw(10) << "Iteration"
        << std::setw(20) << "residual_norm" << std::endl;
    while(1){
        cnt++;
        Vec_t old_u = u;
        for(int t = 0; t < stride; ++t) {
            for(int k = 0; k < N / stride; ++k) {
                int j = k * stride + t;
                Scalar tmp = (A.row(j) * u)(0,0);
                Scalar Ajj = A.coeffRef(j,j);
                u(j) = (u(j) * Ajj - tmp) / Ajj;
            }
        }
        // now u should be GS_op * old_u
        lambda = old_u.dot(u); // Rayleigh quotient
        // compute the residual and check vs tolerance
        auto r = u - lambda * old_u;
        double r_norm = r.norm();
        if(cnt % 50 == 0){
            std::cout << std::left << std::setw(10) << cnt
                << std::setw(20) << r_norm << std::endl;
        }
    
        u.normalize();
       
        if(r_norm < tol) {
            std::cout << "Power iteration for Gauss-Seidel converges after " << cnt 
                << " iterations." << std::endl;
            break;
        }
        if(cnt >= 500) {
            std::cout << "Power iteration for Gauss-Seidel doesn't converge after " << cnt 
                << " iterations." << std::endl; 
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue of Gauss-Seidel by power iteration: " << std::abs(lambda) << std::endl;
    return std::make_pair(u, lambda);
}

void relaxation_factor(PUM_WaveRay& pum_waveray, int L, int num_wavelayer, int wave_number) {

    auto eq_pair = pum_waveray.build_equation(L);
    SpMat_t A(eq_pair.first.makeSparse());
    auto mesh_width = pum_waveray.mesh_width();
    
    std::vector<SpMat_t> Op(num_wavelayer + 1), prolongation_op(num_wavelayer);
    std::vector<int> stride(num_wavelayer + 1);
    Op[num_wavelayer] = A;
    stride[num_wavelayer] = 1;
    for(int i = num_wavelayer - 1; i >= 0; --i) {
        int idx = L + i - num_wavelayer;
        prolongation_op[i] = pum_waveray.prolongation(idx);
        // auto tmp = pum_waveray.build_equation(idx);
        // Op[i] = tmp.first.makeSparse();
        Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
        stride[i] = pum_waveray.num_planwaves[idx];
        // stride[i] = 1;
    }

    // power_waveray(Op[0], 1);

    for(int i = 0; i < Op.size(); ++i) {
        int idx = L + i - num_wavelayer;

        std::cout << "Power iteration of Mesh Operator " << i 
            << ", h = " << mesh_width[idx] << std::endl;
        std::cout << "k*h = " << wave_number * mesh_width[idx] << std::endl;
        power_waveray(Op[i], 1);
        // power_block_GS(Op[i], stride[i]);
        // power_kaczmarz(Op[i]);
    }
} 

int main(){
    std::string square_output = "../result_square/ExtendPUM_WaveRay/";
    std::string square = "../meshes/square.msh";
    std::string square_hole = "../meshes/square_hole.msh";
    size_type L = 3; // refinement steps
 
    double k = 16; // wave number
    std::vector<int> num_planwaves(L+1);
    num_planwaves[L] = 2;
    for(int i = L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }

    plan_wave sol(k, 0.8, 0.6);
    auto u = sol.get_fun();
    auto grad_u = sol.get_gradient();
    auto g = sol.boundary_g();

    // PUM_WaveRay pum_waveray(L, k, square, g, u, false, num_planwaves, 50);
    PUM_WaveRay pum_waveray(L, k, square_hole, g, u, true, num_planwaves, 50);

    int num_wavelayer = 3;
    relaxation_factor(pum_waveray, L, num_wavelayer, k);    
}