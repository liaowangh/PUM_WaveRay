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

void convergence_factor(std::string& mesh, int L, bool hole, std::vector<double> wave_number) {
    std::string output_folder = "../result_square/";
    std::vector<std::vector<double>> factors(wave_number.size());
    std::vector<std::string> data_label;
    for(int k = 0; k < wave_number.size(); ++k) {
        auto wave_nr = wave_number[k];

        plan_wave sol(wave_nr, 0.8, 0.6);
        auto u = sol.get_fun();
        auto grad_u = sol.get_gradient();
        auto g = sol.boundary_g();
        HE_LagrangeO1 he_O1(L, wave_nr, mesh, g, u, hole, 30);
        data_label.push_back(std::to_string(int(wave_nr)));
        for(int i = 0; i <= L; ++i) {
            auto eq_pair = he_O1.build_equation(i);
            Mat_t dense_A = eq_pair.first.makeDense();
            Mat_t A_L = dense_A.triangularView<Eigen::Lower>();
            Mat_t A_U = A_L - dense_A;
            Mat_t GS_op = A_L.colPivHouseholderQr().solve(A_U);
            Vec_t eivals = GS_op.eigenvalues();

            Scalar domainant_eival = eivals(0);
            for(int j = 1; j < eivals.size(); ++j) {
                if(std::abs(eivals(j)) > std::abs(domainant_eival)) {
                    domainant_eival = eivals(j);
                }
            }
            factors[k].push_back(std::abs(domainant_eival));
        }
    }
    std::string str = "GS_ConvergenceFactors";
    tabular_output(factors, data_label, str, output_folder, true);
}

int main() {
    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/ExtendPUM/";
    int L = 2; // refinement steps
    double k = 16;

    std::vector<int> nr_planwaves(L+1, 4);
    plan_wave sol(k, 0.8, 0.6);
    auto u = sol.get_fun();
    auto grad_u = sol.get_gradient();
    auto g = sol.boundary_g();

    HE_ExtendPUM he_epum(L, k, square, g, u, false, nr_planwaves, 20);
    HE_LagrangeO1 he_O1(L, k, square, g, u, false, 20);

    convergence_factor(square, 5, false, {1, 2, 4, 6, 8, 10, 12, 16});
    // auto eq_pair = he_epum.build_equation(L);
    // SpMat_t A = eq_pair.first.makeSparse();
    // int N = A.rows();
    // Vec_t phi = eq_pair.second;
    // Vec_t true_sol = he_epum.solve(L);
    // auto eq_pair = he_O1.build_equation(L);
    // SpMat_t A = eq_pair.first.makeSparse();
    // int N = A.rows();
    // Vec_t phi = eq_pair.second;
    // Vec_t true_sol = he_O1.solve(L);
}