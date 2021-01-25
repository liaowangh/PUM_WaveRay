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

using namespace std::complex_literals;

PUM_WaveRay::FHandle_t exp_wave(Eigen::Vector2d& d, double k) {
    auto res = [&d, &k](const Eigen::Vector2d& x) {
        return std::exp(1i*k * d.dot(x));
    };
    return res;
}

void PUM_WaveRay::Prolongation_Lagrange(){
    P_Lagrange = std::vector<Mat_t>(L);
    for(int l = 0; l < L; ++l) {
        auto coarse_mesh = getmesh(l);
        auto fine_mesh   = getmesh(l+1);
        
        auto coarse_dofh = get_dofh(l);
        auto fine_dof    = get_dofh(l+1);
        
        size_type n_c = coarse_dofh.NumDofs();
        size_type n_f = fine_dof.NumDofs();
        
        Mat_t M(n_c, n_f);
        
        for(const lf::mesh::Entity* edge: fine_mesh->Entities(1)) {
            nonstd::span<const lf::mesh::Entity* const> points = edge->SubEntities(1);
            size_type num_points = (*edge).RefEl().NumSubEntities(1); // number of endpoints, should be 2
            for(int j = 0; j < num_points; ++j) {
                auto parent_p = mesh_hierarchy->ParentEntity(l+1, *points[j]); // parent entity of current point 
                if(parent_p->RefEl() == lf::base::RefEl::kPoint()) {
                    // it's parent is also a NODE, if the point in finer mesh does not show in coarser mesh,
                    // then it's parent is an EDGE
                    M(coarse_mesh->Index(*parent_p), fine_mesh->Index(*points[j])) = 1;
                    M(coarse_mesh->Index(*parent_p), fine_mesh->Index(*points[1-j])) = 0.5;
                }
            }
        }
        P_LF[l] = M.transpose();
    }
}

/*
 * E_l -> E_{l+1}
 * P(e_2t_l) = e_t_{l+1}
 * P(e_{2t+1}_l) = at * e_t_{l+1} + bt * e_{t+1}_{l+1} (get the coefficient by L2 projection)
 */
void PUM_WaveRay::Prolongation_planwave() {
    P_planwave = std::vector<Mat_t>(L);
    
    double domain_area = 0.0;
    for(const lf::mesh::Entity* cell: getmesh(0)->Entity(0)) {
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        domain_area += lf::geometry::Volume(*geo_ptr);
    }

    double pi = std::acos(-1);

    for(int l = 0; l < L; ++l) {
        int N1 = num_planwaves[l];
        int N2 = num_planwaves[l+1];
        
        Mat_t M(N2, N1);
        for(int t = 0; t < N2; ++t) {
            
            Eigen::Vector2d dl1t1, dl1t, d2t1l;

            dl1t1 << std::exp(2*pi*(t+1) / N2), std::sin(2*pi*(t+1) / N2);
            dl1t  << std::exp(2*pi* t    / N2), std::sin(2*pi* t    / N2);
            d2t1l << std::exp(2*pi*(2*t+1)/N2), std::sin(2*pi*(2*t+1)/N2);

            lf::mesh::utils::MeshFunctionGlobal integrand1{exp_wave(dl1t1 - dl1t, k)};
            Scalar A12 = lf::uscalfe::IntegrateMeshFunction(getmesh(L), integrand1, 10);

            Eigen::Matrix<Scalar, 2, 2> A;
            A << domain_area, A12, std::conj(A12), domain_area;
    
            lf::mesh::utils::MeshFunctionGlobal integrand2{exp_wave(d2t1l - dl1t, k)};
            lf::mesh::utils::MeshFunctionGlobal integrand3{exp_wave(d2t1l -dl1t1, k)};
            Eigen::Matrix<Scalar, 2, 1> b;
            b << lf::uscalfe::IntegrateMeshFunction(getmesh(L), integrand2, 10);,
                 lf::uscalfe::IntegrateMeshFunction(getmesh(L), integrand3, 10);;
            auto tmp = A.colPivHouseholderQr().solve(b);
            
            M(t, 2 * t) = 1;
            M(t, 2 * t + 1) = tmp(0);
            M((t+1) / N2, 2 * t + 1) = tmp(1);
        }
        P_Planwave[l] = M;
    }
}

/*
 * E_{l+1} -> E_l
 */
void PUM_WaveRay::Restriction_PW() {
    R_planwave = std::vector<Mat_t>(L);
    int N1 = 1;
    int N2 = 2;
    for(int l = L - 1; l >= 0; --l) {
        N1 = N2;
        N2 = 2*N1;
        R_PW[l] = Mat_t::Zero(N2, N1);
        for(int i = 0; i < N1; ++i){
            R_planwave[l](2*i, i) = 1;
        }
    }
}

// /*
//  * Return the linear operator that map a function from level l to level l+1 
//  */
// PUM_WaveRay::Mat_t PUM_WaveRay::Prolongation_PUM(int l) {
//     LF_ASSERT_MSG((l < L), 
//         "in prolongation, level should smaller than" << L);
//     if(l == L - 1) {
//         auto finest_mesh = mesh_hierarchy->getMesh(L);
//         auto coarser_mesh = mesh_hierarchy->getMesh(L - 1);

//         auto fine_dof = lf::assemble::UniformFEDofHandler(finest_mesh, {{lf::base::RefEl::kPoint(), 1}});
//         auto coarser_dof = lf::assemble::UniformFEDofHandler(coarser_mesh, {{lf::base::RefEl::kPoint(), 1}});

//         size_type n_f = fine_dof.NumDofs();
//         size_type n_c = coarser_dof.NumDofs();

//         size_type n_wave = 4;

//         Mat_t res(n_f, n_wave * n_c);
//         for(size_type t = 0; t < n_wave; ++t) {
//             Mat_t P_L = P_LF[l]; // shape: n_f * n_c
//             //
//             // std::cout << P_L.rows() << " " << P_L.cols() << std::endl;
//             // std::cout << n_f << " " << n_c << " " << std::endl;
//             //
//             auto fre_t = generate_fre(L, k, l, t);
//             auto exp_t = exp_wave(fre_t[0], fre_t[1]);
//             Vec_t exp_at_v(n_f);
//             for(size_type i = 0; i < n_f; ++i) {
//                 const lf::mesh::Entity& v = fine_dof.Entity(i); // the entity to which i-th global shape function is associated
//                 Eigen::Vector2d v_coordinate = lf::geometry::Corners(*v.Geometry()).col(0);
//                 exp_at_v(i) = exp_t(v_coordinate);
//             }
//             res.block(0, t * n_c, n_f, n_c) = exp_at_v.asDiagonal() * P_L; // column wise * or / is not supported
//         }
//         return res;
//     } else {
//         auto Q = P_PW[l];
//         auto P = P_LF[l];
//         size_type n1 = Q.rows(), n2 = P.rows();
//         size_type m1 = Q.cols(), m2 = P.cols();
//         Mat_t res(n1*n2, m1*m2);
//         for(int i = 0; i < n1; ++i) {
//             for(int j = 0; j < m1; ++j) {
//                 res.block(i*n2, j*m2, n2, m2) = Q(i, j) * P;
//             }
//         }
//         return res;
//     }
// }

// /*
//  * Return the linear operator that map a function from level l+1 to level l
//  */
// PUM_WaveRay::Mat_t PUM_WaveRay::Restriction_PUM(int l) {
//     LF_ASSERT_MSG((l < L), 
//         "in prolongation, level should smaller than" << L);
//     if(l == L - 1) {
//         return Prolongation_PUM(l).transpose();
//     } else {
//         auto Q = R_PW[l];
//         auto P = P_LF[l].transpose();
//         size_type n1 = Q.rows(), n2 = P.rows();
//         size_type m1 = Q.cols(), m2 = P.cols();
//         Mat_t res(n1*n2, m1*m2);
//         for(int i = 0; i < n1; ++i) {
//             for(int j = 0; j < m1; ++j) {
//                 res.block(i*n2, j*m2, n2, m2) = Q(i, j) * P;
//             }
//         }
//         return res;
//     }
// }

// template <typename mat_type>
// void Gaussian_Seidel(mat_type& A, PUM_WaveRay::Vec_t& phi, PUM_WaveRay::Vec_t& u, int t) {
//     // u: initial value; t: number of iterations
//     int N = A.rows();
//     for(int i = 0; i < t; ++i){
//         for(int j = 0; j < N; ++j) {
//             auto tmp = A.row(j).dot(u);
//             u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
//         }
//     }
// }

// void PUM_WaveRay::v_cycle(Vec_t& u, size_type mu1, size_type mu2) {
//     auto eq_pair = build_equation(L); // Equaiton Ax=f in finest mesh
//     // const Eigen::SparseMatrix<Scalar> A(eq_pair.first.makeSparse());
//     Mat_t A(eq_pair.first.makeDense());
//     Vec_t f = eq_pair.second;

//     std::vector<Vec_t> initiaLvalue(L + 1);
//     std::vector<Vec_t> rhs_vec(L + 1);
//     // std::vector<Eigen::SparseMatrix<Scalar>> Op(L + 1);
//     std::vector<Mat_t> Op(L + 1);
//     std::vector<Mat_t> restriction(L), prolongation(L);
//     Op[L] = A;
//     initiaLvalue[L] = u;
//     rhs_vec[L] = f;
//     // build coarser mesh operator
//     for(size_type i = L - 1; i >= 0; --i) {
//         restriction[i] = Restriction_PUM(i);
//         prolongation[i] = Prolongation_PUM(i);
//         Op[i] = restriction[i] * Op[i+1] * prolongation[i];
//     }
//     // initial guess on coarser mesh are all zero
//     for(size_type i = L - 1; i >= 0; --i) {
//         initiaLvalue[i] = Vec_t::Zero(Op[i].rows());
//     }
//     Gaussian_Seidel(A, rhs_vec[L], initiaLvalue[L], mu1);  // relaxation mu1 times on finest mesh
//     for(int i = L - 1; i > 0; --i) {
//         rhs_vec[i] = restriction[i] * (rhs_vec[i+1] - Op[i+1] * initiaLvalue[i+1]);
//         Gaussian_Seidel(Op[i], rhs_vec[i], initiaLvalue[i], mu1); 
//     }
//     rhs_vec[0] = restriction[0] * (rhs_vec[1] - Op[1] * initiaLvalue[1]);
//     initiaLvalue[0] = Op[0].colPivHouseholderQr().solve(rhs_vec[0]);
//     for(int i = 1; i <= L; ++i) {
//         initiaLvalue[i] += prolongation[i] * initiaLvalue[i-1];
//         Gaussian_Seidel(Op[i], rhs_vec[i], initiaLvalue[i], mu2);
//     }
//     u = initiaLvalue[L];
// }