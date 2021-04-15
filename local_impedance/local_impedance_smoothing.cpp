#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "local_impedance_solver.h"
#include "../Krylov/GMRES.h"
#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../LagrangeO1/HE_LagrangeO1.h"
#include "../Pum_WaveRay/HE_FEM.h"

using namespace std::complex_literals;

void O1_impedance_vcycle(Vec_t& u, Vec_t& f, HE_LagrangeO1& he_O1,
    std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, int L, 
    double k, std::vector<double> mesh_width) {

    int nu1 = 2, nu2 = 2;
    int n = I.size();
    std::vector<int> op_size(n+1);
    for(int i = 0; i <= n; ++i) {
        op_size[i] = Op[i].rows();
    }
    std::vector<Vec_t> initial(n + 1), rhs_vec(n + 1);

    initial[n] = u;
    rhs_vec[n] = f;
    // initial guess on coarser mesh are all zero
    for(int i = 0; i < n; ++i) {
        initial[i] = Vec_t::Zero(op_size[i]);
    }
    for(int i = n; i > 0; --i) {
        if(k * mesh_width[i] < 1){
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], 1, nu1);
        } else {
            // gmres(Op[i], rhs_vec[i], initial[i], 10, 1);
            int idx = L+i-n;
            vertex_patch_info patch(k, he_O1.getmesh(idx), false, he_O1.innerBdy_selector(idx));
            patch.relaxation(initial[i], rhs_vec[i]);
        }
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }

    Eigen::SparseLU<SpMat_t> solver;
    solver.compute(Op[0]);
    initial[0] = solver.solve(rhs_vec[0]);
   
    for(int i = 1; i <= n; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        if(k * mesh_width[i] < 1.0){
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], 1, nu2);
        } else {
            // gmres(Op[i], rhs_vec[i], initial[i], 20, 1);
            int idx = L+i-n;
            vertex_patch_info patch(k, he_O1.getmesh(idx), false, he_O1.innerBdy_selector(idx));
            patch.relaxation(initial[i], rhs_vec[i]);
        }
    }
    u = initial[n];
}

Vec_t O1_impedance_vcycle(Vec_t f, HE_LagrangeO1& he_O1,
    std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, int L, 
    double k, std::vector<double> mesh_width) {

    int nu1 = 2, nu2 = 2;
    int n = I.size();
    std::vector<int> op_size(n+1);
    for(int i = 0; i <= n; ++i) {
        op_size[i] = Op[i].rows();
    }
    std::vector<Vec_t> initial(n + 1), rhs_vec(n + 1);

    rhs_vec[n] = f;
    // initial guess on coarser mesh are all zero
    for(int i = 0; i <= n; ++i) {
        initial[i] = Vec_t::Zero(op_size[i]);
    }
    for(int i = n; i > 0; --i) {
        if(k * mesh_width[i] < 1){
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], 1, nu1);
        } else {
            // gmres(Op[i], rhs_vec[i], initial[i], 10, 1);
            int idx = L+i-n;
            vertex_patch_info patch(k, he_O1.getmesh(idx), false, he_O1.innerBdy_selector(idx));
            patch.relaxation(initial[i], rhs_vec[i]);
        }
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }

    Eigen::SparseLU<SpMat_t> solver;
    solver.compute(Op[0]);
    initial[0] = solver.solve(rhs_vec[0]);
   
    for(int i = 1; i <= n; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        if(k * mesh_width[i] < 1.0){
            Gaussian_Seidel(Op[i], rhs_vec[i], initial[i], 1, nu2);
        } else {
            // gmres(Op[i], rhs_vec[i], initial[i], 20, 1);
            int idx = L+i-n;
            vertex_patch_info patch(k, he_O1.getmesh(idx), false, he_O1.innerBdy_selector(idx));
            patch.relaxation(initial[i], rhs_vec[i]);
        }
    }
    return initial[n];
}

Vec_t O1_impedance_vcycle(Vec_t f, impedance_smoothing_element& imp) {

    std::vector<SpMat_t>& I = imp.I;
    std::vector<SpMat_t>& Op = imp.Op;
    std::vector<double>& mw = imp.mw;
    int n = I.size();
    std::vector<int> op_size(n+1);
    for(int i = 0; i <= n; ++i) {
        op_size[i] = Op[i].cols();
    }
    std::vector<Vec_t> initial(n + 1), rhs_vec(n + 1);

    rhs_vec[n] = f;
    // initial guess on coarser mesh are all zero
    for(int i = 0; i <= n; ++i) {
        initial[i] = Vec_t::Zero(op_size[i]);
    }
    for(int i = n; i > 0; --i) {
        imp.smoothing(i, initial[i], rhs_vec[i]);
        rhs_vec[i-1] = I[i-1].transpose() * (rhs_vec[i] - Op[i] * initial[i]);
    }

    Eigen::SparseLU<SpMat_t> solver;
    solver.compute(Op[0]);
    initial[0] = solver.solve(rhs_vec[0]);
   
    for(int i = 1; i <= n; ++i) {
        initial[i] += I[i-1] * initial[i-1];
        imp.smoothing(i, initial[i], rhs_vec[i]);
        // skip local smoothing with impedance boundary condition on second leg
        // if(mw[i] * imp.k_ < imp.kh_threshold_){
        //     imp.smoothing(i, initial[i], rhs_vec[i]);
        // }
    }
    return initial[n];
}

void mg_O1_impedance(HE_LagrangeO1& he_O1, int L, int nr_coarsemesh, double k, FHandle_t u) {
    auto eq_pair = he_O1.build_equation(L);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(nr_coarsemesh + 1), prolongation_op(nr_coarsemesh);
    std::vector<double> ms(nr_coarsemesh + 1);
    auto mesh_width = he_O1.mesh_width();
    Op[nr_coarsemesh] = A;
    ms[nr_coarsemesh] = mesh_width[L];
    for(int i = nr_coarsemesh - 1; i >= 0; --i) {
        int idx = L + i - nr_coarsemesh;
        prolongation_op[i] = he_O1.prolongation(idx);
        // Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
        auto tmp = he_O1.build_equation(idx);
        Op[i] = tmp.first.makeSparse();
        ms[i] = mesh_width[idx];
    }

    /**************************************************************************/
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };

    int N = A.rows();
    Vec_t v = Vec_t::Random(N); // initial value
    Vec_t uh = he_O1.solve(L); // finite element solution

    std::vector<double> L2_vk;
    std::vector<double> L2_ek;

    // std::cout << std::scientific << std::setprecision(1);
    for(int i = 0; i < 10; ++i) {
        std::cout << std::setw(15) << he_O1.L2_Err(L, v - uh, zero_fun) << " ";
        std::cout << std::setw(15) << he_O1.L2_Err(L, v, u) << std::endl;
        L2_vk.push_back(he_O1.L2_Err(L, v, zero_fun));
        L2_ek.push_back(he_O1.L2_Err(L, v - uh, zero_fun));
        O1_impedance_vcycle(v, eq_pair.second, he_O1, Op, prolongation_op, L, k, ms);
    }

    std::cout << "||u-uh||_2 = " << he_O1.L2_Err(L, uh, u) << std::endl;
    std::cout << "||v_{i+1}||/||v_i||" << std::endl;
    std::cout << std::left;
    for(int i = 0; i + 1 < L2_vk.size(); ++i) {
        std::cout << i << " " << std::setw(10) << L2_vk[i+1] / L2_vk[i] 
                       << " " << std::setw(10) << L2_ek[i+1] / L2_ek[i] << std::endl;
    }
} 

void KrylovEnhance_impedance(HE_LagrangeO1& he_O1, int L, int nr_coarsemesh, double k, FHandle_t u,
    double relaxation_threshold, int inner_iteration, int outer_iteration) {
    auto eq_pair = he_O1.build_equation(L);
    SpMat_t A(eq_pair.first.makeSparse());

    impedance_smoothing_element imp(he_O1, L, nr_coarsemesh, k, relaxation_threshold);

    Vec_t uh = he_O1.solve(L); // finite element solution
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };
    std::vector<double> L2_vk;
    std::vector<double> L2_ek;

    /*
     * Outer iteration of GMRES, use MG as right preconditioner
     * since GMRES is used as smoothing in certain level of MG and 
     * GMRES is not linear iterations, so we need to use flexible GMRES here.
     */
    int n = A.rows();
    int m = inner_iteration;
    Vec_t x = Vec_t::Random(n); // initial value
    Vec_t b = eq_pair.second;
    double r_norm_initial = (b - A*x).norm();
    std::cout << std::left;
    std::cout << std::setw(5) << "#" << std::setw(15) << "||x-uh||"
              << std::setw(15) << "||r||" << std::setw(15) << "||x-u||" << std::endl;
    for(int t = 0; t < outer_iteration; ++t) {
        std::cout << std::setw(5)  << t;
        std::cout << std::setw(15) << he_O1.L2_Err(L, x - uh, zero_fun);
        std::cout << std::setw(15) << (b - A*x).norm() / r_norm_initial;
        std::cout << std::setw(15) << he_O1.L2_Err(L, x, u) << std::endl;
        L2_vk.push_back(he_O1.L2_Err(L, x, zero_fun));
        L2_ek.push_back(he_O1.L2_Err(L, x - uh, zero_fun));

        Vec_t r = b - A * x;
        double r_norm = r.norm();
        double error = r_norm / r_norm;
        std::vector<double> e;
        e.push_back(error);

        // For givens roatation
        Vec_t sn = Vec_t::Zero(m), cs = Vec_t::Zero(m);
        Vec_t beta = Vec_t::Zero(m+1);
        beta(0) = r_norm;

        Mat_t V(n, m+1);
        V.col(0) = r / r_norm;
        Mat_t Z(n, m);
        Mat_t H = Mat_t::Zero(m+1, m);

        int j;
        for(j = 1; j <= m; ++j) {
            Z.col(j-1) = O1_impedance_vcycle(V.col(j-1), imp);
            Vec_t wj = A * Z.col(j-1);
            for(int i = 0; i < j; ++i) {
                H(i, j-1) = wj.dot(V.col(i));
                wj = wj - H(i, j-1) * V.col(i);
            }
            H(j, j-1) = wj.norm();
            V.col(j) = wj / H(j, j-1);
           
            // eliminate the last element in H jth row and update the rotation matrix
            applay_givens_roataion(H, cs, sn, j-1);
       
            // update the residual vector
            beta(j)  = -sn(j-1) * beta(j-1);
            beta(j-1) = cs(j-1) * beta(j-1);
           
            error = std::abs(beta(j)) / r_norm;
            e.push_back(error);
        }
        // calculate the result
        if(j == m + 1) --j;
        Vec_t y = solve_upper_triangle(H.topLeftCorner(j, j), beta.head(j));
        x = x + Z.leftCols(j) * y;
    }

    std::cout << "||u-uh||_2 = " << he_O1.L2_Err(L, uh, u) << std::endl;
    std::cout << "||v_{i+1}||/||v_i|| ||e_{i+1}||/||e_i||" << std::endl;
    std::cout << std::left;
    for(int i = 0; i + 1 < L2_vk.size(); ++i) {
        std::cout << i << " " << std::setw(10) << L2_vk[i+1] / L2_vk[i] 
                       << " " << std::setw(10) << L2_ek[i+1] / L2_ek[i] << std::endl;
    }
}

int KrylovEnhance_impedance(HE_LagrangeO1& he_O1, int L, int nr_coarsemesh, double k, 
    double relaxation_threshold, double error_threshold, bool verbose=false) {

    auto eq_pair = he_O1.build_equation(L);
    SpMat_t A(eq_pair.first.makeSparse());

    impedance_smoothing_element imp(he_O1, L, nr_coarsemesh, k, relaxation_threshold);

    /*
     * Outer iteration of GMRES, use MG as right preconditioner
     * since GMRES is used as smoothing in certain level of MG and 
     * GMRES is not linear iterations, so we need to use flexible GMRES here.
     */
    int n = A.rows();
    int m = 300;
    Vec_t x = Vec_t::Random(n); // initial value
    Vec_t b = eq_pair.second;
    double r_norm_initial = (b - A*x).norm();
   
    Vec_t r = b - A * x;
    double r_norm = r.norm();
    double error = r_norm / r_norm;
    std::vector<double> e;
    e.push_back(error);

    // For givens roatation
    Vec_t sn = Vec_t::Zero(m), cs = Vec_t::Zero(m);
    Vec_t beta = Vec_t::Zero(m+1);
    beta(0) = r_norm;

    Mat_t V(n, m+1);
    V.col(0) = r / r_norm;
    Mat_t Z(n, m);
    Mat_t H = Mat_t::Zero(m+1, m);

    int j;
    if(verbose) {
        std::cout << "m  ||rm||" << std::endl;
    }
    for(j = 1; j <= m; ++j) {
        Z.col(j-1) = O1_impedance_vcycle(V.col(j-1), imp);
        Vec_t wj = A * Z.col(j-1);
        for(int i = 0; i < j; ++i) {
            H(i, j-1) = wj.dot(V.col(i));
            wj = wj - H(i, j-1) * V.col(i);
        }
        H(j, j-1) = wj.norm();
        V.col(j) = wj / H(j, j-1);
        
        // eliminate the last element in H jth row and update the rotation matrix
        applay_givens_roataion(H, cs, sn, j-1);
    
        // update the residual vector
        beta(j)  = -sn(j-1) * beta(j-1);
        beta(j-1) = cs(j-1) * beta(j-1);
        
        error = std::abs(beta(j)) / r_norm;
        e.push_back(error);
        if(verbose) std::cout << j << "  " << error << std::endl;
        if(error < error_threshold) {
            break; 
        }
    }
    // calculate the result
    if(j == m + 1) --j;
    Vec_t y = solve_upper_triangle(H.topLeftCorner(j, j), beta.head(j));
    x = x + Z.leftCols(j) * y;
    return j;
}

void iteration_count() {
    std::string output = "../result/impedance_count/";
    std::vector<std::string> meshes{"../meshes/square.msh", "../meshes/square_hole2.msh", "../meshes/triangle_hole.msh"};
    // std::vector<std::string> meshes{"../meshes/square_hole2.msh"};
    std::vector<bool> hole{false, true, true};
    // std::vector<double> wave_number{4, 6, 8, 10, 12, 14, 16, 18, 20};
    std::vector<int> wave_number{10, 12, 14, 16, 18, 20};
    int L = 4;
    int num_coarserlayer = L;

    double error_threshold = 0.000001;
    double relaxation_threshold = 1.5;

    int nr_mesh = meshes.size();
    std::vector<std::vector<int>> counts(nr_mesh+1);
    counts[nr_mesh] = wave_number;
    for(int mesh_idx = 0; mesh_idx < nr_mesh; ++mesh_idx) {
        for(int k_idx = 0; k_idx < wave_number.size(); ++k_idx) {
            double k = wave_number[k_idx];
            plan_wave sol(k, 0.8, 0.6);
            auto u = sol.get_fun();
            auto g = sol.boundary_g();

            HE_LagrangeO1 he_O1(L, k, meshes[mesh_idx], g, u, hole[mesh_idx], 50);
            int it_count = KrylovEnhance_impedance(he_O1, L, num_coarserlayer, k, relaxation_threshold, error_threshold, true);
            counts[mesh_idx].push_back(it_count);
        }
        std::cout << meshes[mesh_idx] << " finished" << std::endl;
    } 
    std::vector<std::string> mesh_name{"unit_square", "square_hole", "triangle_hole", "k"};
    // std::vector<std::string> mesh_name{"square_hole", "k"};
    std::string str = "gs_above_8_L" + std::to_string(L);
    print_save_error(counts, mesh_name, str, output);
}

int main(){
    std::string square = "../meshes/square.msh";
    std::string square_hole2 = "../meshes/square_hole2.msh";
    std::string triangle_hole = "../meshes/triangle_hole.msh";

    iteration_count();
    // size_type L = 5; // refinement steps
    // double k = 30.0; // wave number
    // plan_wave sol(k, 0.8, 0.6);
    // auto u = sol.get_fun();
    // auto g = sol.boundary_g();
    // auto grad_u = sol.get_gradient();
    // HE_LagrangeO1 he_O1(L, k, square, g, u, false, 50);
    // // HE_LagrangeO1 he_O1(L, k, square_hole2, g, u, true, 50);
    // // HE_LagrangeO1 he_O1(L, k, triangle_hole, g, u, true, 50);
    
    // int num_coarserlayer = L;
    // // mg_O1_impedance(he_O1, L, num_coarserlayer, k, u);    
    // KrylovEnhance_impedance(he_O1, L, num_coarserlayer, k, u, 1.5, 20, 20);
    // KrylovEnhance_impedance(he_O1, L, num_coarserlayer, k, 1.5, 0.000001, true);
    return 0;
}