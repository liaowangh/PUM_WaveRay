#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../Pum_WaveRay/PUM_WaveRay.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

/*
 * True solution u = exp(ikx)
 * Fine mesh equation: Au=f
 * Let the fine approximation vh = 0, then eh = exp(ikx), and residual r = f
 * Transfer the r to the next coarse PUM space W_{L-1}
 * do the relaxation (or solve it), then transfer it back to fine grid.
 */
int main(){
    std::string square = "../meshes/square.msh";
    std::string square_hole2 = "../meshes/square_hole2.msh";
    std::string triangle_hole = "../meshes/triangle_hole.msh";

    size_type L = 5; // refinement steps
    double k = 20.0; // wave number
    std::vector<int> num_planwaves(L+1);
    num_planwaves[L] = 2;
    for(int i = L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }

    auto zero_fun = [](const coordinate_t& x)->Scalar { return 0.0; };
    plan_wave sol(k, 0.8, 0.6);
    auto u = sol.get_fun();
    auto grad_u = sol.get_gradient();
    auto g = sol.boundary_g();

    PUM_WaveRay pum(L, k, square, g, u, false, num_planwaves, 50);
    Mat_t P1 = Mat_t(pum.prolongation(L-1)); // prolongation operator: L-1 -> L
    Mat_t P2 = Mat_t(pum.prolongation(L-2));
    auto eq_pair1 = pum.build_equation(L);
    auto eq_pair2 = pum.build_equation(L-1);
    Mat_t A1 = eq_pair1.first.makeDense();
    Vec_t f = eq_pair1.second; // rhs, also the residual if vh = 0
    Mat_t A2 = P1.transpose() * A1 * P1;
    Mat_t A3 = P2.transpose() * A2 * P2;
    // Mat_t A2 = eq_pair2.first.makeDense();
    auto uh = pum.solve(L);
    Vec_t vh, vH;
    // vh = P1 * P2 * A3.colPivHouseholderQr().solve(P2.transpose() * P1.transpose() * f);
    vh = P1 * A2.colPivHouseholderQr().solve(P1.transpose() * f);
    // std::cout << ((P1.transpose() * f) - eq_pair2.second).norm() << std::endl;
    std::cout << "||uh-u||  : " << pum.L2_Err(L, uh, u) << std::endl;
    std::cout << "||uh-vh|| : " << pum.L2_Err(L, uh - vh, zero_fun) << std::endl;
    std::cout << "||vh-u||  : " << pum.L2_Err(L, vh, u) << std::endl;
    // for(int i = 0; i < vH.size(); ++i) {
    //     std::cout << i << " " << vH(i) << std::endl;
    // }
}