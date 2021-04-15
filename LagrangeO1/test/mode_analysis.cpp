#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/filesystem.hpp>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../../utils/HE_solution.h"
#include "../../utils/utils.h"
#include "../HE_LagrangeO1.h"

using namespace std::complex_literals;

void O1_vcycle(HE_LagrangeO1& he_O1, int start_layer, int nr_coarselayer) {
    auto eq_pair = he_O1.build_equation(start_layer);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(nr_coarselayer + 1), prolongation_op(nr_coarselayer);
    std::vector<int> stride(nr_coarselayer + 1, 1);
    Op[nr_coarselayer] = A;
    for(int i = nr_coarselayer - 1; i >= 0; --i) {
        int idx = start_layer + i - nr_coarselayer;
        prolongation_op[i] = he_O1.prolongation(idx);
        auto tmp = he_O1.build_equation(idx);
        Op[i] = tmp.first.makeSparse();
    }

    Eigen::MatrixXd A_real = A.real(), A_imag = A.imag();
    Eigen::EigenSolver<Eigen::MatrixXd> es_real(A_real), es_imag(A_imag);

    

    Vec_t zero_vec = Vec_t::Zero(A.rows());
    for(int i = 0; i < A_ev.cols(); ++i) {
        Vec_t ev = A_ev.col(i);
        Vec_t initial = A_ev.col(i);
        for(int nu = 0; nu < 10; ++nu){
            v_cycle(initial, zero_vec, Op, prolongation_op, stride, 3, 3, false);
        }
        std::cout << i << " " << initial.norm() / ev.norm() << std::endl; 
    }
}

int main(){
    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/LagrangeO1/";
    size_type L = 5; // refinement steps
    double k = 6.31; // wave number
   
    plan_wave solution(k, 0.8, 0.6); 
   
    auto u = solution.get_fun();
    auto g = solution.boundary_g();
    auto grad_u = solution.get_gradient();
    HE_LagrangeO1 he_O1(L, k, square, g, u, false, 20);

    int nr_coarselayer = 3;
    O1_vcycle(he_O1, L, nr_coarselayer);
}