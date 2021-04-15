#pragma once

#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../Krylov/GMRES.h"
#include "../Pum_WaveRay/HE_FEM.h"

using Scalar = std::complex<double>;
using size_type = unsigned int;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using SpMat_t = Eigen::SparseMatrix<Scalar>;

class mg_element {
public:
    mg_element(HE_FEM& he_fem, int L, int nr_coarselayers) {
        auto eq_pair = he_fem.build_equation(L);
        SpMat_t A(eq_pair.first.makeSparse());
        auto mesh_width = he_fem.mesh_width();
        op = std::vector<SpMat_t>(nr_coarselayers + 1);
        I = std::vector<SpMat_t>(nr_coarselayers);
        mw = std::vector<double>(nr_coarselayers + 1);
        block_size = std::vector<int>(nr_coarselayers + 1);

        op[nr_coarselayers] = A;
        mw[nr_coarselayers] = mesh_width[L];
        block_size[nr_coarselayers] = he_fem.Dofs_perNode(L);

        for(int i = nr_coarselayers - 1; i >= 0; --i) {
            int idx = L + i - nr_coarselayers;
            I[i] = he_fem.prolongation(idx);
            op[i] = I[i].transpose() * op[i+1] * I[i];
            mw[i] = mesh_width[idx];
            block_size[i] = he_fem.Dofs_perNode(idx);
        }
        phi = eq_pair.second;
    }
public:
    std::vector<SpMat_t> op;  // mesh operator
    std::vector<SpMat_t> I;   // prolongation operator
    std::vector<int> block_size; // used in PUM spaces
    std::vector<double> mw;   // mesh_width
    Vec_t phi;
};

void ray_cycle(Vec_t& v, Vec_t& phi, const mg_element& ray_mg, double k, bool solve_coarest=true) {
    const std::vector<SpMat_t>& I = ray_mg.I;
    std::vector<SpMat_t> Op = ray_mg.op;
    const std::vector<int>& block_size = ray_mg.block_size;
    const std::vector<double> ms = ray_mg.mw;

    int nu1 = 2, nu2 = 2;
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
        if(k * ms[i] < 1){
            block_GS(Op[i], rhs_vec[i], initial[i], block_size[i], nu1);
        } else {
            // gmres(Op[i], rhs_vec[i], initial[i], 10, 1);
            // Kaczmarz(Op[i], rhs_vec[i], initial[i], 5 * nu1);
        }
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }
    // std::cout << "Finisth first leg of ray cycle." << std::endl;
    if(solve_coarest) {
        Eigen::SparseLU<SpMat_t> solver;
        solver.compute(Op[0]);
        initial[0] = solver.solve(rhs_vec[0]);
    } else {
        if(k * ms[0] < 1){
            block_GS(Op[0], rhs_vec[0], initial[0], block_size[0], nu1 + nu2);
        } else {
            // gmres(Op[0], rhs_vec[0], initial[0], 20, 1);
        }
    }

    for(int i = 1; i <= nr_raylayer; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        if(k * ms[i] < 1){
            block_GS(Op[i], rhs_vec[i], initial[i], block_size[i], nu2);
        } else {
            // gmres(Op[i], rhs_vec[i], initial[i], 20, 1);
            // Kaczmarz(Op[i], rhs_vec[i], initial[i], 5 * nu1);
        }
    }
    v = initial[nr_raylayer];
} 

void wave_cycle(Vec_t& v, Vec_t& phi, const mg_element& wave_mg, const mg_element& ray_mg,
    double k, int wave_L, int ray_L) {

    const std::vector<SpMat_t>& wave_I = wave_mg.I;
    std::vector<SpMat_t> wave_op = wave_mg.op;
    const std::vector<int>& block_size = wave_mg.block_size;
    const std::vector<double> wave_ms = wave_mg.mw;

    int nu1 = 2, nu2 = 2;
    int wave_coarselayers = wave_I.size();
    std::vector<int> op_size(wave_coarselayers + 1);
    for(int i = 0; i <= wave_coarselayers; ++i) {
        op_size[i] = wave_op[i].rows();
    }
    std::vector<Vec_t> initial(wave_coarselayers + 1), rhs_vec(wave_coarselayers + 1);

    /************ first leg of wave cycle *************/
    initial[wave_coarselayers] = v;
    rhs_vec[wave_coarselayers] = phi;
    // initial guess on coarser mesh are all zero
    for(int i = 0; i < wave_coarselayers; ++i) {
        initial[i] = Vec_t::Zero(op_size[i]);
    }
    for(int i = wave_coarselayers; i > 0; --i) {
        if(k * wave_ms[i] < 1.5 || k * wave_ms[i] > 8.0){
            Gaussian_Seidel(wave_op[i], rhs_vec[i], initial[i], 1, nu1);
        } else {
            // Kaczmarz(wave_op[i], rhs_vec[i], initial[i], 5 * nu1);
            // gmres(wave_op[i], rhs_vec[i], initial[i], 10, 1);
        }
        rhs_vec[i-1] = wave_I[i-1].transpose() * (rhs_vec[i] - wave_op[i] * initial[i]);
    }

    Eigen::SparseLU<SpMat_t> solver;
    solver.compute(wave_op[0]);
    initial[0] = solver.solve(rhs_vec[0]);

    /************ second leg of wave cycle *************/
    for(int i = 1; i <= wave_coarselayers; ++i) {
        initial[i] += wave_I[i-1] * initial[i-1];
        if(wave_L + i - wave_coarselayers == ray_L){
            ray_cycle(initial[i], rhs_vec[i], ray_mg, k, true);
            // ray_cycle(initial[i], rhs_vec[i], ray_mg, k, true);
        } else if(k * wave_ms[i] < 1.5 || k * wave_ms[i] > 8.0){
            Gaussian_Seidel(wave_op[i], rhs_vec[i], initial[i], 1, nu2);
        } else {
            // gmres(wave_op[i], rhs_vec[i], initial[i], 20, 1);
            // Kaczmarz(wave_op[i], rhs_vec[i], initial[i], 5 * nu1);
        }
    }
    v = initial[wave_coarselayers];
    /* finish one iteration of wave-ray */   
}