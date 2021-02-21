#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "ExtendPUM_WaveRay.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

std::pair<lf::assemble::COOMatrix<ExtendPUM_WaveRay::Scalar>, ExtendPUM_WaveRay::Vec_t> 
ExtendPUM_WaveRay::build_equation(size_type level) {
    return level == L ? HE_LagrangeO1::build_equation(level) :
                        HE_ExtendPUM::build_equation(level);
}

double ExtendPUM_WaveRay::L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) {
    return l == L ? HE_LagrangeO1::L2_Err(l, mu, u) :
                    HE_ExtendPUM::L2_Err(l, mu, u);
}

double ExtendPUM_WaveRay::H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) {
    return l == L ? HE_LagrangeO1::H1_semiErr(l, mu, grad_u):
                    HE_ExtendPUM::H1_semiErr(l, mu, grad_u);
}

double ExtendPUM_WaveRay::H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u){
    double l2err = L2_Err(l, mu, u);
    double h1serr = H1_semiErr(l, mu, grad_u);
    return std::sqrt(l2err * l2err + h1serr * h1serr);
}

ExtendPUM_WaveRay::Vec_t ExtendPUM_WaveRay::fun_in_vec(size_type l, const FHandle_t& f) {
    return l == L ? HE_LagrangeO1::fun_in_vec(l, f) :
                    HE_ExtendPUM::fun_in_vec(l, f);
}

ExtendPUM_WaveRay::Vec_t ExtendPUM_WaveRay::solve(size_type l) {
    return l == L ? HE_LagrangeO1::solve(l) : HE_ExtendPUM::solve(l);
}

ExtendPUM_WaveRay::SpMat_t ExtendPUM_WaveRay::prolongation(size_type l) {
    return l == L - 1 ? HE_ExtendPUM::prolongation_SE_S() : HE_ExtendPUM::prolongation(l);
    // if(l == L -1) {
    //     return HE_ExtendPUM::prolongation_SE_S();
    // }
    // double pi = std::acos(-1.);
    // auto Q = prolongation_lagrange(l);
    // int n1 = Q.cols(), n2 = Q.rows(); // n1: n_l, n2: n_{l+1}
    // int N1 = num_planwaves[l], N2 = num_planwaves[l+1]; // N1: N_l, N2: N_{l+1}

    // auto mesh = getmesh(l+1);  // fine mesh
    // auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});

    // SpMat_t res(n2*(N2+1), n1*(N1+1)); // transfer operator
    // std::vector<triplet_t> triplets;
    
    // for(int outer_idx = 0; outer_idx < Q.outerSize(); ++outer_idx) {
    //     for(SpMat_t::InnerIterator it(Q, outer_idx); it; ++it) {
    //         int i = it.row();
    //         int j = it.col();
    //         Scalar qij = it.value(); 
            
    //         // b_j^l = \sum_i qij b_i^{l+1} (t = 0)
    //         triplets.push_back(triplet_t(i*(N2+1), j*(N1+1), qij));

    //         // b_j^l e_{2t-1}^l = \sum_i qij b_i^{l+1} e_t^{l+1}
    //         for(int t = 1; t <= N2; ++t) {
    //             triplets.push_back(triplet_t(i*(N2+1)+t, j*(N1+1)+2*t-1, qij));
    //         } 

    //         const lf::mesh::Entity& p_i = dofh.Entity(i); // the entity to which i-th global shape function is associated
    //         coordinate_t pi_coordinate = lf::geometry::Corners(*p_i.Geometry()).col(0);

    //         for(int t = 1; t <= N2; ++t) {
    //             Eigen::Vector2d d1, d2;
    //             d1 << std::cos(2*pi*(2*t-1)/N1), std::sin(2*pi*(2*t-1)/N1); // d_{2t}^l
    //             d2 << std::cos(2*pi*(  t-1)/N2), std::sin(2*pi*(  t-1)/N2); // d_{t}^{l+1}
    //             Scalar tmp = qij * std::exp(1i*k*(d1-d2).dot(pi_coordinate));
    //             triplets.push_back(triplet_t(i*(N2+1)+t, j*(N1+1)+2*t, tmp));
    //         }

    //     }
    // }
    // res.setFromTriplets(triplets.begin(), triplets.end());
    // return res;
}

/*
 * v-cycle, mu1, mu2 -- pre and post smoothing times
 */
ExtendPUM_WaveRay::Vec_t ExtendPUM_WaveRay::solve_multigrid(size_type start_layer, int num_wavelayer,
    int mu1, int mu2) {
    // in pum waveray, only start_layer = L makes sense.
    LF_ASSERT_MSG((num_wavelayer <= start_layer), 
        "please use a smaller number of wave layers");
    auto eq_pair = build_equation(start_layer);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(num_wavelayer + 1), prolongation_op(num_wavelayer);
    std::vector<int> stride(num_wavelayer + 1);
    Op[num_wavelayer] = A;
    stride[num_wavelayer] = 1;
    for(int i = num_wavelayer - 1; i >= 0; --i) {
        int idx = start_layer + i - num_wavelayer;
        // auto tmp = build_equation(idx);
        // Op[i] = tmp.first.makeSparse();
        prolongation_op[i] = prolongation(idx);
        Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
        stride[i] = num_planwaves[idx] + 1;
    }
    Vec_t initial = Vec_t::Random(A.rows());
    // Vec_t initial = HE_LagrangeO1::solve_multigrid(start_layer, 3, 5, 5);
    v_cycle(initial, eq_pair.second, Op, prolongation_op, stride, mu1, mu2);
    return initial;
}

std::pair<ExtendPUM_WaveRay::Vec_t, ExtendPUM_WaveRay::Scalar> 
ExtendPUM_WaveRay::power_multigird(size_type start_layer, int num_coarserlayer, 
        int mu1, int mu2) {

    LF_ASSERT_MSG((num_coarserlayer <= start_layer), 
        "please use a smaller number of wave layers");
    auto eq_pair = build_equation(start_layer);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
    std::vector<int> stride(num_coarserlayer + 1);
    Op[num_coarserlayer] = A;
    stride[num_coarserlayer] = 1;
    for(int i = num_coarserlayer - 1; i >= 0; --i) {
        int idx = start_layer + i - num_coarserlayer;
        prolongation_op[i] = prolongation(idx);
        // auto tmp = build_equation(idx);
        // Op[i] = tmp.first.makeSparse();
        Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
        stride[i] = num_planwaves[idx] + 1;
    }

    int N = A.rows();
    
    /* Get the multigrid (2 grid) operator manually */
    if(num_coarserlayer == 1) {
        Mat_t coarse_op = Mat_t(Op[0]);
        Mat_t mg_op = Mat_t::Identity(N, N) - 
            prolongation_op[0]*coarse_op.colPivHouseholderQr().solve(Mat_t(prolongation_op[0]).transpose())*Op[1];

        Mat_t L = Mat_t(A.triangularView<Eigen::Lower>());
        Mat_t U = L - A;
        Mat_t GS_op = L.colPivHouseholderQr().solve(U);

        Mat_t R_mu1 = Mat_t::Identity(N, N);
        Mat_t R_mu2 = Mat_t::Identity(N, N);
        for(int i = 0; i < mu1; ++i) {
            auto tmp = R_mu1 * GS_op;
            R_mu1 = tmp;
        }
        for(int i = 0; i < mu2; ++i) {
            auto tmp = R_mu2 * GS_op;
            R_mu2 = tmp;
        }
        auto tmp = R_mu2 * mg_op * R_mu1;
        mg_op = tmp;

        Vec_t eivals = mg_op.eigenvalues();

        Scalar domainant_eival = eivals(0);
        for(int i = 1; i < eivals.size(); ++i) {
            if(std::abs(eivals(i)) > std::abs(domainant_eival)) {
                domainant_eival = eivals(i);
            }
        }

        // std::string output_file = "../plot_err/eigenvalues/k2";
        // std::ofstream out(output_file);
        // if(out) {
        //     out << "EigenValues" << std::endl;
        //     out << eivals.cwiseAbs();
        // } else {
        //     std::cout << "Cannot open file " << output_file << std::endl;
        // }

        std::cout << eivals << std::endl;
        std::cout << "Domainant eigenvalue: " << domainant_eival << std::endl;
        std::cout << "Absolute value: " << std::abs(domainant_eival) << std::endl;
    }
    /***************************************/

    Vec_t u = Vec_t::Random(N);
    u.normalize();
    Vec_t old_u;
    Vec_t zero_vec = Vec_t::Zero(N);
    Scalar lambda;
    int cnt = 0;
    
    std::cout << std::left << std::setw(10) << "Iteration" 
        << std::setw(20) << "residual_norm" << std::endl;
    while(true) {
        cnt++;
        old_u = u;
        v_cycle(u, zero_vec, Op, prolongation_op, stride, mu1, mu2);
        // u = mg_op * old_u;
        
        lambda = old_u.dot(u);  // domainant eigenvalue
        auto r = u - lambda * old_u;
        
        u.normalize();
    
        if(cnt % 1 == 0) {
            std::cout << std::left << std::setw(10) << cnt 
                << std::setw(20) << r.norm() 
                << std::setw(20) << (u - old_u).norm()
                << std::endl;
        }
        if(r.norm() < 0.01) {
            break;
        }
        if(cnt > 20) {
            std::cout << "Power iteration for multigrid doesn't converge." << std::endl;
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue by power iteration: " << lambda << std::endl;
    vector_vtk(start_layer, u, "square_L3_k2pi_2mesh_Galerkin");
    return std::make_pair(u, lambda);
}