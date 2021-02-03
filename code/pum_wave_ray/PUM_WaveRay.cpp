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

#include "PUM_WaveRay.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

PUM_WaveRay::FHandle_t exp_wave(const Eigen::Vector2d& d, double k) {
    auto res = [&d, &k](const Eigen::Vector2d& x) {
        return std::exp(1i*k * d.dot(x));
    };
    return res;
}

std::pair<lf::assemble::COOMatrix<PUM_WaveRay::Scalar>, PUM_WaveRay::Vec_t> 
PUM_WaveRay::build_equation(size_type level) {
    return level == L ? HE_LagrangeO1::build_equation(level) :
                        HE_PUM::build_equation(level);
}

double PUM_WaveRay::L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) {
    return l == L ? HE_LagrangeO1::L2_Err(l, mu, u) :
                    HE_PUM::L2_Err(l, mu, u);
}

double PUM_WaveRay::H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) {
    return l == L ? HE_LagrangeO1::H1_semiErr(l, mu, grad_u):
                    HE_PUM::H1_semiErr(l, mu, grad_u);
}

double PUM_WaveRay::H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u){
    double l2err = L2_Err(l, mu, u);
    double h1serr = H1_semiErr(l, mu, grad_u);
    return std::sqrt(l2err * l2err + h1serr * h1serr);
}

PUM_WaveRay::Vec_t PUM_WaveRay::fun_in_vec(size_type l, const FHandle_t& f) {
    return l == L ? HE_LagrangeO1::fun_in_vec(l, f) :
                    HE_PUM::fun_in_vec(l, f);
}

// S_l -> S_{l+1}， 0 <= l < L
void PUM_WaveRay::Prolongation_Lagrange(){
    P_Lagrange = std::vector<Mat_t>(L);
    for(int l = 0; l < L; ++l) {
        auto coarse_mesh = getmesh(l);
        auto fine_mesh   = getmesh(l+1);
        
        auto coarse_dofh = lf::assemble::UniformFEDofHandler(coarse_mesh, 
                           {{lf::base::RefEl::kPoint(), 1}});
        auto fine_dof    = lf::assemble::UniformFEDofHandler(fine_mesh, 
                           {{lf::base::RefEl::kPoint(), 1}});
        
        size_type n_c = coarse_dofh.NumDofs();
        size_type n_f = fine_dof.NumDofs();
        
        Mat_t M(n_c, n_f);

        for(const lf::mesh::Entity* edge: fine_mesh->Entities(1)) {
            nonstd::span<const lf::mesh::Entity* const> points = edge->SubEntities(1);
            size_type num_points = (*edge).RefEl().NumSubEntities(1); // number of endpoints, should be 2
            LF_ASSERT_MSG((num_points == 2), 
                "Every EDGE should have 2 kPoint subentities");
            for(int j = 0; j < num_points; ++j) {
                auto parent_p = mesh_hierarchy->ParentEntity(l+1, *points[j]); // parent entity of current point 
                if(parent_p->RefEl() == lf::base::RefEl::kPoint()) {
                    // it's parent is also a NODE. If the point in finer mesh does not show in coarser mesh,
                    // then it's parent is an EDGE
                    M(coarse_mesh->Index(*parent_p), fine_mesh->Index(*points[j])) = 1;
                    M(coarse_mesh->Index(*parent_p), fine_mesh->Index(*points[1-j])) = 0.5;
                }
            }
        }
        P_Lagrange[l] = M.transpose();
    }
}

/*
 * E_l -> E_{l+1}, l = 0, 1, ..., L-1
 * P(e_2t_l) = e_t^{l+1}
 * P(e_{2t+1}_l) = at * e_t_{l+1} + bt * e_{t+1}_{l+1} (get the coefficient by L2 projection)
 * e_t^l = exp(ik(cos(2pi*t/N)x + sin(2pi*t/N)y)) t = 0, 1, ..., N-1 (note that in my thesis, t starts from 1)
 */
void PUM_WaveRay::Prolongation_planwave() {
    P_planwave = std::vector<Mat_t>(L);
    auto mesh = getmesh(0);
    double domain_area = 0.0;
    for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        domain_area += lf::geometry::Volume(*geo_ptr);
    }

    mesh = getmesh(L);
    double pi = std::acos(-1);

    for(int l = 0; l < L; ++l) {
        int N1 = num_planwaves[l];
        int N2 = num_planwaves[l+1];

        Mat_t M(N2, N1);
        for(int t = 0; t < N2; ++t) {
            
            Eigen::Vector2d dl1_t1, dl1_t, dl_2t1;

            dl1_t1 << std::cos(2*pi*(t+1) / N2), std::sin(2*pi*(t+1) / N2); // e^{l+1}_{t+1}
            dl1_t  << std::cos(2*pi* t    / N2), std::sin(2*pi* t    / N2); // e^{l+1}_t
            dl_2t1 << std::cos(2*pi*(2*t+1)/N1), std::sin(2*pi*(2*t+1)/N1); // e^l_{2t+1}

            auto f1 = [&dl1_t1, &dl1_t, this](const Eigen::Vector2d& x)->Scalar {
                return std::exp(1i*k*(dl1_t1 - dl1_t).dot(x));
            };

            lf::mesh::utils::MeshFunctionGlobal mf_f1{f1};
            
            Scalar A12 = lf::uscalfe::IntegrateMeshFunction(*mesh, mf_f1, 10);

            Eigen::Matrix<Scalar, 2, 2> A;
            A << domain_area, A12, std::conj(A12), domain_area;
    
            auto f2 = [&dl_2t1, &dl1_t, this](const Eigen::Vector2d& x)->Scalar {
                return std::exp(1i*k*(dl_2t1 - dl1_t).dot(x));
            };

            auto f3 = [&dl_2t1, &dl1_t1, this](const Eigen::Vector2d& x)->Scalar {
                return std::exp(1i*k*(dl_2t1 - dl1_t1).dot(x));
            };

            lf::mesh::utils::MeshFunctionGlobal mf_f2{f2};
            lf::mesh::utils::MeshFunctionGlobal mf_f3{f3};
            Eigen::Matrix<Scalar, 2, 1> b;
            b << lf::uscalfe::IntegrateMeshFunction(*mesh, mf_f2, 10),
                 lf::uscalfe::IntegrateMeshFunction(*mesh, mf_f3, 10);
            auto tmp = A.colPivHouseholderQr().solve(b);

            M(t, 2 * t) = 1;
            M(t, 2 * t + 1) = tmp(0);
            M((t+1) % N2, 2 * t + 1) = tmp(1);
        }
        P_planwave[l] = M;
    }
}

/*
 * SxE_l -> SxE_{l+1}, is the kronecker product of P_Lagrange[l] and P_planwave[l]
 */
void PUM_WaveRay::Prolongation_SE() {
    // LF_ASSERT_MSG((l < L), 
    //     "in prolongation, level should smaller than" << L);
    P_SE = std::vector<Mat_t>(L);
    for(int l = 0; l < L; ++l) {
        auto Q = P_Lagrange[l];
        auto P = P_planwave[l];
        size_type n1 = Q.rows(), n2 = P.rows();
        size_type m1 = Q.cols(), m2 = P.cols();
        Mat_t res(n1*n2, m1*m2);
        for(int i = 0; i < n1; ++i) {
            for(int j = 0; j < m1; ++j) {
                res.block(i*n2, j*m2, n2, m2) = Q(i, j) * P;
            }
        }
        P_SE[l] = res;
    }
}

/*
 * SxE_{L-1} -> S_L
 */
PUM_WaveRay::Mat_t PUM_WaveRay::Prolongation_SE_S() {
    double pi = std::acos(-1.);
    auto Q = P_Lagrange[L-1];
    int n = Q.rows();
    int N = num_planwaves[L-1];

    Mat_t E(n, N);

    auto S_mesh = getmesh(L);
    auto S_dofh = lf::assemble::UniformFEDofHandler(S_mesh, 
                        {{lf::base::RefEl::kPoint(), 1}});
    for(int i = 0; i < n; ++i) {
        const lf::mesh::Entity& vi = S_dofh.Entity(i); // the entity to which i-th global shape function is associated
        coordinate_t vi_coordinate = lf::geometry::Corners(*vi.Geometry()).col(0);
        for(int t = 0; t < N; ++t) {
            Eigen::Vector2d dt;
            dt << std::cos(2*pi*t/N), std::cos(2*pi*t/N);
            E(i,t) = std::exp(1i*k*dt.dot(vi_coordinate));
        }
    }
    Mat_t res(n, Q.cols() * N);
    // for(int i = 0; i < n; ++i) {
    //     for(int j = 0; j < Q.cols(); ++j) {
    //         res.block(i, j*N, 1, N) = Q(i,j) * E.row(i);
    //     }
    // }
    for(int j = 0; j < Q.cols(); ++j) {
        res.block(0, j*N, n, N) = Q.col(j).asDiagonal() * E;
    }
    return res;
}

/*
 * v-cycle, mu1, mu2 -- pre and post smoothing times
 */
void PUM_WaveRay::solve(Vec_t& initial, int num_wavelayer, int mu1, int mu2) {
    LF_ASSERT_MSG((num_wavelayer <= L), 
        "please use a smaller number of wave layers");
    auto eq_pair = build_equation(L);
    Mat_t A(eq_pair.first.makeDense());

    std::vector<Mat_t> Op(num_wavelayer + 1), prolongation_op(num_wavelayer);
    std::vector<int> stride(num_wavelayer + 1);
    Op[num_wavelayer] = A;
    stride[num_wavelayer] = 1;
    for(int i = num_wavelayer - 1; i >= 0; --i) {
        int idx = L + i - num_wavelayer;
        auto tmp = build_equation(idx);
        Op[i] = tmp.first.makeDense();
        if(i == num_wavelayer - 1) {
            prolongation_op[i] = Prolongation_SE_S();
        } else {
            prolongation_op[i] = P_SE[idx];
        }
        stride[idx] = num_planwaves[idx];
    }

    v_cycle(initial, eq_pair.second, Op, prolongation_op, stride, mu1, mu2);
}