#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../LagrangeO1/HE_LagrangeO1.h"
#include "../ExtendPum/HE_ExtendPUM.h"

using Scalar = std::complex<double>;
using size_type = unsigned int;
using coordinate_t = Eigen::Vector2d;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using triplet_t = Eigen::Triplet<Scalar>;
using SpMat_t = Eigen::SparseMatrix<Scalar>;
using FHandle_t = std::function<Scalar(const coordinate_t&)>;
using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

using namespace std::complex_literals;

int main() {
    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/ExtendPUM/";
    int L = 3; // refinement steps
    double k = 1;

    std::vector<int> nr_planwaves(L+1, 4);
    plan_wave sol(k, 0.8, 0.6);
    auto u = sol.get_fun();
    auto grad_u = sol.get_gradient();
    auto g = sol.boundary_g();

    HE_ExtendPUM he_epum(L, k, square, g, u, false, nr_planwaves, 20);
    HE_LagrangeO1 he_O1(L, k, square, g, u, false, 20);

    auto eq_pair = he_epum.build_equation(L);
    SpMat_t A = eq_pair.first.makeSparse();
    int N = A.rows();
    Vec_t phi = eq_pair.second;
    Vec_t true_sol = he_epum.solve(L);
    // auto eq_pair = he_O1.build_equation(L);
    // SpMat_t A = eq_pair.first.makeSparse();
    // int N = A.rows();
    // Vec_t phi = eq_pair.second;
    // Vec_t true_sol = he_O1.solve(L);

    // Vec_t GS_sol = Vec_t::Random(N);
    // Vec_t block_GS_sol = Vec_t::Random(N);
    // Vec_t Kaczmarz_sol = Vec_t::Random(N);
    // Gaussian_Seidel(A, phi, GS_sol, true_sol, nr_planwaves[L] + 1);
    // block_GS(A, phi, block_GS_sol, true_sol, nr_planwaves[L] + 1);
    // Kaczmarz(A, phi, Kaczmarz_sol, true_sol);

    power_GS(A, nr_planwaves[L] + 1);
    power_block_GS(A, nr_planwaves[L] + 1);
    power_kaczmarz(A);
}