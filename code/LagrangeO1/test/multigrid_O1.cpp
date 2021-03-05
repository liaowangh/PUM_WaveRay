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

void O1_vcycle(Vec_t& u, Vec_t& f, std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, 
    std::vector<int>& stride, double k, std::vector<double> mesh_width, 
    size_type mu1, size_type mu2, bool solve_on_coarest) {

    int L = I.size();
    LF_ASSERT_MSG(Op.size() == L + 1 && stride.size() == L + 1, 
        "#{transfer operator} should be #{Operator} - 1");
    
    std::vector<int> op_size(L+1);
    for(int i = 0; i <= L; ++i) {
        op_size[i] = Op[i].rows();
    }

    for(int i = 0; i < L; ++i) {
        LF_ASSERT_MSG(I[i].rows() == op_size[i+1] && I[i].cols() == op_size[i],
            "transfer operator size does not mathch grid operator size.");
    }

    std::vector<Vec_t> initial(L + 1), rhs_vec(L + 1);

    initial[L] = u;
    rhs_vec[L] = f;
    // initial guess on coarser mesh are all zero
    for(int i = 0; i < L; ++i) {
        initial[i] = Vec_t::Zero(op_size[i]);
    }
    for(int i = L; i > 0; --i) {
        if(k * mesh_width[i] < 2.0 || k * mesh_width[i] > 6.0){
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], mu1);
            // block_GS(Op[i], rhs_vec[i], initial[i], stride[i], mu1);
        } else {
            Kaczmarz(Op[i], rhs_vec[i], initial[i], 5 * mu1);
        }
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }

    if(solve_on_coarest) {
        Eigen::SparseLU<SpMat_t> solver;
        solver.compute(Op[0]);
        initial[0] = solver.solve(rhs_vec[0]);
    } else {
        if(k * mesh_width[0] < 2.0 || k * mesh_width[0] > 6.0){
            Gaussian_Seidel(Op[0], rhs_vec[0], initial[0], stride[0], mu1);
            // block_GS(Op[i], rhs_vec[i], initial[i], stride[i], mu1);
        } else {
            Kaczmarz(Op[0], rhs_vec[0], initial[0], 5 * mu1);
        }
    }
    for(int i = 1; i <= L; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        if(k * mesh_width[i] < 2.0 || k * mesh_width[i] > 6.0){
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], stride[i], mu1);
            // block_GS(Op[i], rhs_vec[i], initial[i], stride[i], mu1);
        } else {
            Kaczmarz(Op[i], rhs_vec[i], initial[i], 5 * mu1);
        }
    }
    u = initial[L];
}

void mg_O1(HE_LagrangeO1& he_O1, int L, int nr_coarsemesh, double k, FHandle_t u, bool solve_coarest) {
    auto eq_pair = he_O1.build_equation(L);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(nr_coarsemesh + 1), prolongation_op(nr_coarsemesh);
    std::vector<int> stride(nr_coarsemesh + 1, 1);
    std::vector<double> ms(nr_coarsemesh + 1);
    auto mesh_width = he_O1.mesh_width();
    Op[nr_coarsemesh] = A;
    ms[nr_coarsemesh] = mesh_width[L];
    for(int i = nr_coarsemesh - 1; i >= 0; --i) {
        int idx = L + i - nr_coarsemesh;
        prolongation_op[i] = he_O1.prolongation(idx);
        auto tmp = he_O1.build_equation(idx);
        Op[i] = tmp.first.makeSparse();
        // Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
        ms[i] = mesh_width[idx];
    }

    // for(int i = 0; i < ms.size(); ++i) {
    //     int idx = i + L - nr_coarsemesh;
    //     std::cout << "h_" << idx << " = " << ms[i] << std::endl; 
    //     Mat_t dense_A = Mat_t(Op[i]);
    //     Mat_t A_L = dense_A.triangularView<Eigen::Lower>();
    //     Mat_t A_U = A_L - dense_A;
    //     Mat_t GS_op = A_L.colPivHouseholderQr().solve(A_U);
    //     Vec_t eivals = GS_op.eigenvalues();

    //     Scalar domainant_eival = eivals(0);
    //     for(int j = 1; j < eivals.size(); ++j) {
    //         if(std::abs(eivals(j)) > std::abs(domainant_eival)) {
    //             domainant_eival = eivals(j);
    //         }
    //     }
    //     std::cout << "Domainant eigenvalue: " << domainant_eival << std::endl;
    //     std::cout << "Absolute value: " << std::abs(domainant_eival) << std::endl;
    //     power_GS(Op[i], 1);
    // }
    /**************************************************************************/
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };

    auto dofh = he_O1.get_dofh(L);
    int N = dofh.NumDofs();

    Vec_t v = Vec_t::Random(N); // initial value
    Vec_t uh = he_O1.solve(L); // finite element solution

    int nu1 = 1, nu2 = 1;

    std::vector<double> L2_vk;
    std::vector<double> L2_ek;

    // std::cout << std::scientific << std::setprecision(1);
    for(int k = 0; k < 10; ++k) {
        std::cout << std::setw(15) << he_O1.L2_Err(L, v - uh, zero_fun) << " ";
        std::cout << std::setw(15) << he_O1.L2_Err(L, v, u) << std::endl;
        // std::cout << v.norm() << " " << std::endl;
        L2_vk.push_back(he_O1.L2_Err(L, v, zero_fun));
        L2_ek.push_back(he_O1.L2_Err(L, v - uh, zero_fun));
        // he_O1.solve_multigrid(v, L, nr_coarsemesh, nu1, nu2, solve_coarest);
        O1_vcycle(v, eq_pair.second, Op, prolongation_op, stride, k, ms, nu1, nu2, solve_coarest);
    }

    std::cout << "||u-uh||_2 = " << he_O1.L2_Err(L, uh, u) << std::endl;
    std::cout << "||v_{k+1}||/||v_k||" << std::endl;
    std::cout << std::left;
    for(int k = 0; k + 1 < L2_vk.size(); ++k) {
        std::cout << k << " " << std::setw(10) << L2_vk[k+1] / L2_vk[k] 
                       << " " << std::setw(10) << L2_ek[k+1] / L2_ek[k] << std::endl;
    }
} 

std::pair<Vec_t, Scalar> power_GS(SpMat_t A) {
    double tol = 0.0001;
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
        for(int j = 0; j < N; ++j) {
            Scalar tmp = (A.row(j) * u)(0,0);
            Scalar Ajj = A.coeffRef(j,j);
            u(j) = (u(j) * Ajj - tmp) / Ajj;
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
    std::cout << "Domainant eigenvalue of Gauss-Seidel by power iteration: " << lambda << std::endl;
    return std::make_pair(u, lambda);
}

void convergence_factor(HE_LagrangeO1& he_O1, int L, double k) {

    std::vector<SpMat_t> Op(L+1);
    for(int l = 0; l <= L; ++l) {
        auto eq_pair = he_O1.build_equation(l);
        Op[l] = eq_pair.first.makeSparse();
    }
    std::vector<double> mesh_width = he_O1.mesh_width();
    // std::vector<std::vector<double>> factors(Op.size());   
    for(int i = 0; i < Op.size(); ++i) {
        auto ei_pair = power_GS(Op[i]);
        // factors[i].push_back(std::abs(domainant_eival));
        std::cout << "k*h = " << k * mesh_width[i] 
            << ", convergence factor = " << std::abs(ei_pair.second) << std::endl;
    }

}

int main(){
    // mesh path
    // boost::filesystem::path here = __FILE__;
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/square.msh")).string();

    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/LagrangeO1/";
    std::string square_hole = "../meshes/square_hole.msh";
    size_type L = 5; // refinement steps
    double k = 16; // wave number

    plan_wave sol(k, 0.8, 0.6);
    auto u = sol.get_fun();
    auto g = sol.boundary_g();
    auto grad_u = sol.get_gradient();
    HE_LagrangeO1 he_O1(L, k, square, g, u, false, 20);
    // HE_LagrangeO1 he_O1(L, k, square_hole, g, u, true, 20);
    // convergence_factor(he_O1, L, k);
    
    // int num_coarserlayer = 4;
    // mg_O1(he_O1, L, num_coarserlayer, k, u, true);

    // auto mesh_width = he_O1.mesh_width();
    // for(int i = 0; i < mesh_width.size(); ++i) {
    //     std::cout << i << " " << mesh_width[i] * k << std::endl;
    // }
    auto eq_pair = he_O1.build_equation(3);
    Mat_t A = eq_pair.first.makeSparse();
    Mat_t L_A = Mat_t(A.triangularView<Eigen::Lower>());
    Mat_t U_A = L_A - A;
    Mat_t GS_op = L_A.colPivHouseholderQr().solve(U_A);
    Vec_t eivals = GS_op.eigenvalues();

    Scalar domainant_eival = eivals(0);
    for(int i = 1; i < eivals.size(); ++i) {
        if(std::abs(eivals(i)) > std::abs(domainant_eival)) {
            domainant_eival = eivals(i);
        }
    }
    std::cout << eivals << std::endl;
    std::cout << "Diagonal entriy of A: " << std::endl;
    for(int i = 0; i < A.rows(); ++i){
        std::cout << i << " " << std::abs(A(i,i)) << std::endl; 
    }
}