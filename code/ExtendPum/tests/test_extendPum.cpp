#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../HE_ExtendPUM.h"
#include "../ExtendPUM_ElemVector.h"
#include "../../utils/HE_solution.h"
#include "../../utils/utils.h"

using coordinate_t = Eigen::Vector2d;
using Scalar = std::complex<double>;
using size_type = unsigned int;
using FHandle_t = std::function<Scalar(const coordinate_t&)>;
using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;
using namespace std::complex_literals;

/*
 * Plan waves: u(x) = exp(ik(d1*x(0) + d2*x(1)) + 1/k^2 satisfies \laplac u + k^2u = 1
 * k is the wave number, and frequency d1^2+d2^2 = 1
 * 
 * grad u = ik * u(x) * [d1, d1]
 */
class plan_wave_c: public HE_sol {
public:
    plan_wave_c(double k_, double d1_, double d2_): HE_sol(k_), d1(d1_), d2(d2_){}

    FHandle_t get_fun() {
        return [this](const coordinate_t& x)-> Scalar {
            return std::exp(1i * k * (d1 * x(0) + d2 * x(1))) + 1. / (k * k);
        };
    }

    FunGradient_t get_gradient() {
        return [this](const coordinate_t& x) {
            auto tmp = 1i * k * std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
            Eigen::Matrix<Scalar, 2, 1> res;
            res << tmp * d1, tmp * d2;
            return res;
        };
    }
    
    FHandle_t boundary_g() {
        auto g = [this](const coordinate_t& x) -> Scalar {
            double x1 = x(0), y1 = x(1);
            Scalar res = 1i * k * std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
            if(y1 == 0 && x1 <= 1 && x1 >= 0) {
                // (0,-1)
                res *= (-d2 - 1);
            } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
                // (1,0)
                res *= (d1 - 1);
            } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
                //(0, 1)
                res *= (d2 - 1);
            } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
                // (-1,0)
                res *= (-d1 - 1);
            } 
            return res - 1i / k;
        };
    return g;
    }
private:
    double d1;
    double d2;
};

Vec_t solve_special_he(HE_ExtendPUM& he_epum, int l, int k, int degree);

void output_result(std::vector<std::vector<double>>& data, 
    std::vector<std::string>& data_label, const std::string& sol_name) {
    
    std::cout << sol_name << std::endl;
    //Tabular output of the data
    // std::cout << std::left << std::setw(10) << data_label[i];
    std::cout << std::left;
    for(int i = 0; i < data_label.size(); ++i){
        std::cout << std::setw(10) << data_label[i];
    }
    std::cout << std::endl;
    std::cout << std::left << std::scientific << std::setprecision(1);
    for(int l = 0; l < data[0].size(); ++l) {
        std::cout << l << " & ";
        for(int i = 0; i < data.size(); ++i) {
            std::cout << std::setw(10) << data[i][l];
            if(i == data.size() - 1) {
                std::cout << " \\\\";
            } else {
                std::cout << " & ";
            }
        }
        std::cout << std::endl;
    }
}

/*
 * Use the extend PUM space to approximate the solution of 
 * \laplace u + k^2u = 1
 * grad u n - iku = g on \Gamma_R
 * u = h on \Gamma_D
 */
int main() {
    std::string square_hole = "../meshes/square_hole.msh";
    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/non-homogeneous_HE/";
    int L = 4; // refinement steps
    std::vector<int> number_waves{5, 7, 9, 11};
    std::vector<double> wave_number{6, 20};

    for(auto k: wave_number) {
        plan_wave_c sol(k, 0.8, 0.6);
        std::string str = "k" + std::to_string(int(k));

        auto u = sol.get_fun();
        auto grad_u = sol.get_gradient();
        auto g = sol.boundary_g();

        std::vector<std::vector<double>> L2err(number_waves.size()), H1serr(number_waves.size());
        std::vector<std::string> data_label;
        for(int i = 0; i < number_waves.size(); ++i) {
            int nr_waves = number_waves[i];
            data_label.push_back(std::to_string(nr_waves));
            HE_ExtendPUM he_epum(L, k, square, g, u, false, std::vector<int>(L+1, nr_waves), 20);

            for(int l = 0; l <= L; ++l) {
                Vec_t fe_sol = solve_special_he(he_epum, l, k, 20);
                double l2_err = he_epum.L2_Err(l, fe_sol, u);
                double h1_serr = he_epum.H1_semiErr(l, fe_sol, grad_u);
                
                L2err[i].push_back(l2_err);
                H1serr[i].push_back(h1_serr);
            }
        }
        // output_result(L2err, data_label, str + "_L2");
        // output_result(H1serr, data_label, str + "_H1serr");
        tabular_output(L2err, data_label, str + "_L2err", square_output, true);
        tabular_output(H1serr, data_label, str + "_H1serr", square_output, true);
    }
}

/*
 * solve for \laplace u + k^2 u = 1
 * \int_\Omega \bar(v)dx should be added to right hand side vector 
 */
Vec_t solve_special_he(HE_ExtendPUM& he_epum, int l, int k, int degree = 20) {
    auto eq_pair = he_epum.build_equation(l);
    const Eigen::SparseMatrix<Scalar> A_crs(eq_pair.first.makeSparse());
    Vec_t rhs = eq_pair.second;

    // assemble for \int_\Omega \bar(v) dx
    auto f = [](const coordinate_t& x)-> Scalar {
        return -1.0;
    };

    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(he_epum.getmesh(l));
    
    size_type N_wave(he_epum.Dofs_perNode(l) - 1);
    auto dofh = he_epum.get_dofh(l);

    // assemble for \int (1,v) dx
    ExtendPUM_ElemVec elvec_builder(fe_space, N_wave, k, f, degree);
    lf::assemble::AssembleVectorLocally(0, dofh, elvec_builder, rhs);

    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A_crs);
    Vec_t fe_sol;
    if(solver.info() == Eigen::Success) {
        fe_sol = solver.solve(rhs);
    } else {
        LF_ASSERT_MSG(false, "Eigen Factorization failed");
    }
    return fe_sol;
}