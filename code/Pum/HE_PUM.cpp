#include <vector>
#include <functional>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "HE_PUM.h"
#include "PUM_ElementMatrix.h"
#include "PUM_EdgeMat.h"
#include "PUM_EdgeVector.h"
#include "PUM_ElemVector.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

/*
 * Build equation A*x = phi in PUM spaces
 */
std::pair<lf::assemble::COOMatrix<HE_PUM::Scalar>, HE_PUM::Vec_t> 
HE_PUM::build_equation(size_type level) {
    auto mesh = getmesh(level);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    size_type N_wave(num_planewaves_[level]);
    auto dofh = get_dofh(level);
    size_type N_dofs(dofh.NumDofs());

    // assemble for <grad(u), grad(v)> - k^2 <u,v>
    // (u, v) -> \int_K \alpha * (grad u, grad v) + \gamma * (u, v) dx
    PUM_ElementMatrix elmat_builder(N_wave, k_, 1., -1. * k_ * k_, degree_);

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    auto outer_boundary{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)};
    if(hole_exist_) {
        // assemble boundary edge matrix, -i*k*u*v over \Gamma_R (outer boundary)
        // first need to distinguish between outer and inner boundar
        auto outer_nr = reader_->PhysicalEntityName2Nr("outer_boundary");
        auto inner_nr = reader_->PhysicalEntityName2Nr("inner_boundary");

        // modify it to classify inner and outer boundary
        for(const lf::mesh::Entity* edge: mesh->Entities(1)) {
            if(outer_boundary(*edge)) {
                // find a boundary edge, need to determine if it's outer boundary
                const lf::mesh::Entity* parent_edge = edge;
                for(int i = level; i > 0; --i) {
                    parent_edge = mesh_hierarchy_->ParentEntity(i, *parent_edge);
                }
                if(reader_->IsPhysicalEntity(*parent_edge, inner_nr)) {
                    // it is the inner boundary
                    outer_boundary(*edge) = false;
                }
            }
        }
    }

    // (u,v) -> \int_e gamma * (u,v) dS
    PUM_EdgeMat edge_mat_builder(fe_space, outer_boundary, N_wave, k_, -1i * k_, degree_);                                  
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector, \int_{\Gamma_R} gv.conj dS
    Vec_t phi(N_dofs);
    phi.setZero();
    // l(v) = \int_e g * v.conj dS(x)
    PUM_EdgeVec edgeVec_builder(fe_space, outer_boundary, N_wave, k_, g_, degree_);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);

    if(hole_exist_) {
        // Treatment of Dirichlet boundary conditions h = u|_{\Gamma_D} (inner boundary condition)
        // flag all nodes on the boundary
        auto inner_point{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 2)};
        // flag all nodes on the inner boundary
        for(const lf::mesh::Entity* edge: mesh->Entities(1)) {
            if(outer_boundary(*edge)) {
                // mark the points associated with outer boundary edge to false
                for(const lf::mesh::Entity* subent: edge->SubEntities(1)) {
                    inner_point(*subent) = false;
                }
            }
        }

        // Set up predicate: Run through all global shape functions and check whether
        // they are associated with an entity on the boundary, store Dirichlet data.
        // A L2 projection is used to get Dirichlet data

        auto inner_boundary{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)};
        // modify it to classify inner and outer boundary
        for(const lf::mesh::Entity* edge: mesh->Entities(1)) {
            if(outer_boundary(*edge)) {
                inner_boundary(*edge) = false;
            }
        }
      
        auto h_vec = fun_in_vec(level, h_);
        // auto h_vec_2 = h_in_vec(level, inner_boundary, inner_point);

        // if(level <= 1){
        //     std::cout << L2_BoundaryErr(level, h_vec, h_, inner_boundary) << std::endl;
        //     std::cout << L2_BoundaryErr(level, h_vec_2, h_, inner_boundary) << std::endl;
        //     std::cout << h_vec << std::endl << std::endl;
        //     std::cout << h_vec_2 << std::endl;
        // }

        std::vector<std::pair<bool, Scalar>> ess_dof_select{};
        for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
            const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
            const Scalar h_val = h_vec(dofnum);
            // const Scalar h_val = h_vec_2(dofnum);
            if(inner_point(dof_node)) {
                // Dof associated with an entity on the boundary: "essential dof"
                // The value of the dof should be set to the correspoinding value of the 
                // vector representation of h_
                ess_dof_select.push_back({true, h_val});
            } else {
                ess_dof_select.push_back({false, h_val});
            }
        }

        // modify linear system of equations
        lf::assemble::FixFlaggedSolutionCompAlt<Scalar>(
            [&ess_dof_select](size_type dof_idx)->std::pair<bool, Scalar> {
                return ess_dof_select[dof_idx];},
        A, phi);
    }

    return std::make_pair(A, phi);
}

/*
 * To get the vector representation of function f w.r.t the PUM space,
 * a L2 projection is used.
 * P2(f) = uh, such that ||f-uh|| is minimized, which is equivalent to 
 * find uh such that (uh, v) = (f,v) for all v in the PUM space
 */
HE_PUM::Vec_t HE_PUM::fun_in_vec(size_type l, const FHandle_t& f) {
    auto mesh = mesh_hierarchy_->getMesh(l);  // get mesh
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    size_type N_wave(num_planewaves_[l]);
    auto dofh = get_dofh(l);
    size_type N_dofs(dofh.NumDofs());

    // assemble for \int u * v.conj dx
    PUM_ElementMatrix elmat_builder(N_wave, k_, 0, 1., degree_);

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);

    // assemble for \int (f,v) dx
    Vec_t phi(N_dofs);
    phi.setZero();
    PUM_ElemVec elvec_builder(fe_space, N_wave, k_, f, degree_);
    lf::assemble::AssembleVectorLocally(0, dofh, elvec_builder, phi);

    const Eigen::SparseMatrix<Scalar> A_crs(A.makeSparse());

    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A_crs);
    Vec_t res;
    if(solver.info() == Eigen::Success) {
        res = solver.solve(phi);
    } else {
        LF_ASSERT_MSG(false, "Eigen Factorization failed");
    }
    return res;
}

/*
 * To get the vector representation of function h in the Dirichlet boundary
 */
HE_PUM::Vec_t HE_PUM::h_in_vec(size_type l, lf::mesh::utils::CodimMeshDataSet<bool> edge_selector,
    lf::mesh::utils::CodimMeshDataSet<bool> inner_point) {
    auto mesh = mesh_hierarchy_->getMesh(l);  // get mesh
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    size_type N_wave(num_planewaves_[l]);
    auto dofh = get_dofh(l);
    size_type N_dofs(dofh.NumDofs());

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);

    // assemble for \int_e (u,v) dx 
    PUM_EdgeMat edge_mat_builder(fe_space, edge_selector, N_wave, k_, 1.0, degree_);                                  
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);

    // assemble for \int_e (h,v) dx
    Vec_t phi(N_dofs);
    phi.setZero();
    PUM_EdgeVec edgeVec_builder(fe_space, edge_selector, N_wave, k_, h_, degree_);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);

    std::vector<std::pair<bool, Scalar>> ess_dof_select{};
        for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
            const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
            if(inner_point(dof_node)) {
                ess_dof_select.push_back({false, 0.0});
            } else {
                ess_dof_select.push_back({true, 0.0});
            }
        }

    lf::assemble::FixFlaggedSolutionCompAlt<Scalar>(
        [&ess_dof_select](size_type dof_idx)->std::pair<bool, Scalar> {
            return ess_dof_select[dof_idx];}, 
    A, phi);

    const Eigen::SparseMatrix<Scalar> A_crs(A.makeSparse());

    // if(l == 0) {
    //     std::cout << A_crs << std::endl;
    //     std::cout << phi << std::endl;
    // }

    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A_crs);
    Vec_t res;
    if(solver.info() == Eigen::Success) {
        res = solver.solve(phi);
    } else {
        LF_ASSERT_MSG(false, "Eigen Factorization failed");
    }
    return res;
}

// quadrature based norm computation for ||uh-u||
double HE_PUM::L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) {
    auto mesh = mesh_hierarchy_->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

    size_type N_wave(num_planewaves_[l]);
    auto dofh = get_dofh(l);

    for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
    
        // Matrix storing corner coordinates in its columns(2x3 in this case)
        auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
        
        // suppose that the barycentric coordinate functions have the form
        // \lambda_i = a + b1*x+b2*y, then
        // \lambda_i = X(0,i) + X(1,i)*x + X(2,i)*y
        // grad \lambda_i = [X(1,i), X(2,i)]^T
        // grad \lambda_1_2_3 = X.block<2,3>(1,0)
        Eigen::Matrix3d X, tmp;
        tmp.block<3,1>(0,0) = Eigen::Vector3d::Ones();
        tmp.block<3,2>(0,1) = vertices.transpose();
        X = tmp.inverse();

        // Number of shape functions covering current entity
        const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*cell));
        //obtain global indices of those shape functions
        nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};
        // then uh(FE solution) restricted in the cell is \sum_i \lambda_(i/N_wave)*exp(ik d_{i%N_wave} x) * mu(dofarray(i))

        auto integrand = [this, &mu, &u, &X, &N_wave, &dofarray](const Eigen::Vector2d& x)->Scalar {
            Scalar val_uh = 0.0;
            Scalar val_u  = u(x);
            for(int i = 0; i < 3; ++i) {
                for(int t = 0; t < N_wave; ++t) {
                    Eigen::Matrix<Scalar, 2, 1> d;
                    double pi = std::acos(-1);
                    d << std::cos(2*pi*t/N_wave), std::sin(2*pi*t/N_wave);
                    val_uh += mu(dofarray[i*N_wave+t]) * (X(0,i) + X(1,i)*x(0) + X(2,i)*x(1)) * std::exp(1i*k_*d.dot(x));
                }
            }
            return std::abs((val_uh-val_u)*(val_uh-val_u));
        };
        res += std::abs(LocalIntegral(*cell, degree_, integrand));
    }
    return std::sqrt(res);
}

double HE_PUM::H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) {
    auto mesh = mesh_hierarchy_->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

    size_type N_wave(num_planewaves_[l]);
    auto dofh = get_dofh(l);

    for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
    
        // Matrix storing corner coordinates in its columns(2x3 in this case)
        auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
        
        // suppose that the barycentric coordinate functions have the form
        // \lambda_i = a + b1*x+b2*y, then
        // \lambda_i = X(0,i) + X(1,i)*x + X(2,i)*y
        // grad \lambda_i = [X(1,i), X(2,i)]^T
        // grad \lambda_1_2_3 = X.block<2,3>(1,0)
        Eigen::Matrix3d X, tmp;
        tmp.block<3,1>(0,0) = Eigen::Vector3d::Ones();
        tmp.block<3,2>(0,1) = vertices.transpose();
        X = tmp.inverse();

        // Number of shape functions covering current entity
        const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*cell));
        //obtain global indices of those shape functions
        nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};
        // then uh(FE solution) restricted in the cell is \sum_i \lambda_(i/N_wave)*exp(ik d_{i%N_wave} x) * mu(dofarray(i))
        // construct ||grad uh - grad u||^2_{cell}

        auto integrand = [this, &mu, &grad_u, &X, &N_wave, &dofarray](const Eigen::Vector2d& x)->Scalar {
            // for f(x) = \lambda_i * exp(ikdt x)
            // grad f(x) = exp(ikdt x) * beta_i + ik*exp(ikdt x) * \lambda_i * dt
            Eigen::Matrix<Scalar, 2, 1> val_grad_uh, val_grad_u;
            val_grad_uh << 0.0, 0.0;
            val_grad_u = grad_u(x);
            for(int i = 0; i < 3; ++i) {
                for(int t = 0; t < N_wave; ++t) {
                    Eigen::Matrix<Scalar, 2, 1> d;
                    Eigen::Vector2d beta;
                    double pi = std::acos(-1);
                    d << std::cos(2*pi*t/N_wave), std::sin(2*pi*t/N_wave);
                    beta << X(1,i), X(2,i);
                    double lambda = X(0,i) + beta.dot(x);
                    val_grad_uh += mu(dofarray[i*N_wave+t]) * std::exp(1i*k_*d.dot(x)) * (beta + 1i*k_*lambda*d);
                }
            }
            return std::abs((val_grad_uh - val_grad_u).dot((val_grad_uh - val_grad_u)));
        };
        res += std::abs(LocalIntegral(*cell, degree_, integrand));
    }
    return std::sqrt(res);
}

double HE_PUM::H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) {
    double l2err = L2_Err(l, mu, u);
    double h1serr = H1_semiErr(l, mu, grad_u);
    return std::sqrt(l2err*l2err + h1serr*h1serr);
}

// compute \int_{\partial Omega} |mu-u|^2dS
double HE_PUM::L2_BoundaryErr(size_type l, const Vec_t& mu, const FHandle_t& u,
    lf::mesh::utils::CodimMeshDataSet<bool> edge_selector) {
    auto mesh = mesh_hierarchy_->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

    size_type N_wave(num_planewaves_[l]);
    auto dofh = get_dofh(l);

    for(const lf::mesh::Entity* cell: mesh->Entities(1)) {
        if(edge_selector(*cell)){
            const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        
            // Matrix storing corner coordinates in its columns(2x2 in this case)
            auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
            
            //obtain global indices of those shape functions
            nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};
            // then uh(FE solution) restricted in the cell (line) is \sum_i \lambda_(i/N_wave)*exp(ik d_{i%N_wave} x) * mu(dofarray(i))

            auto integrand = [this, &mu, &u, &vertices, &N_wave, &dofarray](const Eigen::Vector2d& x)->Scalar {
                Scalar val_uh = 0.0;
                Scalar val_u  = u(x);
                double x1, y1, x2, y2, tmp;
                x1 = vertices(0,0), y1 = vertices(1,0);
                x2 = vertices(0,1), y2 = vertices(1,1);
                if(x1 == x2) {
                    tmp = (x(1) - y2) / (y1 - y2);
                } else {
                    tmp = (x(0) - x2) / (x1 - x2); 
                }
                for(int t = 0; t < N_wave; ++t) {
                    Eigen::Matrix<double, 2, 1> d;
                    double pi = std::acos(-1.);
                    d << std::cos(2*pi*t/N_wave), std::sin(2*pi*t/N_wave);
                    val_uh += (mu(dofarray[t]) * tmp + mu(dofarray[t+N_wave]) * (1-tmp)) * std::exp(1i*k_*d.dot(x));
                }
                return std::abs((val_uh-val_u)*(val_uh-val_u));
            };
            res += std::abs(LocalIntegral(*cell, degree_, integrand));
        }
    }
    return std::sqrt(res);
}

HE_PUM::Vec_t HE_PUM::solve(size_type l) {
    auto eq_pair = build_equation(l);
    const Eigen::SparseMatrix<Scalar> A_crs(eq_pair.first.makeSparse());
    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A_crs);
    Vec_t fe_sol;
    if(solver.info() == Eigen::Success) {
        fe_sol = solver.solve(eq_pair.second);
    } else {
        LF_ASSERT_MSG(false, "Eigen Factorization failed");
    }
    return fe_sol;
}

/*
 * SxE_l -> SxE_{l+1}, 
 *  is the kronecker product of prolongation_lagrange(l) and prolongation_planwave(l)
 */
HE_PUM::SpMat_t HE_PUM::prolongation(size_type l) {
    LF_ASSERT_MSG((l < L_), 
        "in prolongation, level should smaller than" << L_);
    
    double pi = std::acos(-1.);
    auto Q = prolongation_lagrange(l);
    int n1 = Q.cols(), n2 = Q.rows(); // n1: n_l, n2: n_{l+1}
    int N1 = num_planewaves_[l], N2 = num_planewaves_[l+1]; // N1: N_l, N2: N_{l+1}

    auto mesh = getmesh(l+1);  // fine mesh
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});

    SpMat_t res(n2 * N2, n1 * N1); // transfer operator
    std::vector<triplet_t> triplets;
    
    for(int outer_idx = 0; outer_idx < Q.outerSize(); ++outer_idx) {
        for(SpMat_t::InnerIterator it(Q, outer_idx); it; ++it) {
            int i = it.row();
            int j = it.col();
            Scalar qij = it.value(); 
            
            // b_j^l e_{2t-1}^l = \sum_i qij b_i^{l+1} e_t^{l+1}
            for(int t = 1; t <= N2; ++t) {
                triplets.push_back(triplet_t(i*N2+t-1, j*N1+2*t-2, qij));
            } 

            const lf::mesh::Entity& p_i = dofh.Entity(i); // the entity to which i-th global shape function is associated
            coordinate_t pi_coordinate = lf::geometry::Corners(*p_i.Geometry()).col(0);

            for(int t = 1; t <= N2; ++t) {
                Eigen::Vector2d d1, d2;
                d1 << std::cos(2*pi*(2*t-1)/N1), std::sin(2*pi*(2*t-1)/N1); // d_{2t}^l
                d2 << std::cos(2*pi*(  t-1)/N2), std::sin(2*pi*(  t-1)/N2); // d_{t}^{l+1}
                Scalar tmp = qij * std::exp(1i*k_*(d1-d2).dot(pi_coordinate));
                triplets.push_back(triplet_t(i*N2+t-1, j*N1+2*t-1, tmp));
            }
        }
    }
    res.setFromTriplets(triplets.begin(), triplets.end());
    return res;

    // auto Q = prolongation_lagrange(l);
    // auto P = prolongation_planwave(l);
    // size_type n1 = Q.rows(), n2 = P.rows();
    // size_type m1 = Q.cols(), m2 = P.cols();
    // SpMat_t res(n1*n2, m1*m2);
    // std::vector<triplet_t> triplets;
    // for(int j1 = 0; j1 < Q.outerSize(); ++j1) {
    //     for(SpMat_t::InnerIterator it1(Q, j1); it1; ++it1) {
    //         int i1 = it1.row();
    //         Scalar qij = it1.value();
    //         for(int j2 = 0; j2 < P.outerSize(); ++j2) {
    //             for(SpMat_t::InnerIterator it2(P, j2); it2; ++it2) {
    //                 int i2 = it2.row();
    //                 Scalar pij = it2.value();
    //                 triplets.push_back(triplet_t(i1*n2+i2, j1*m2+j2, qij * pij));
    //             }
    //         }
    //     }
    // }
    // res.setFromTriplets(triplets.begin(), triplets.end());
    // return res;
}

void HE_PUM::solve_multigrid(Vec_t& initial, size_type start_layer, int num_coarserlayer, 
    int nu1, int nu2, bool solve_coarest) {

    LF_ASSERT_MSG((num_coarserlayer <= L_), 
        "please use a smaller number of wave layers");
    auto eq_pair = build_equation(L_);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
    std::vector<int> stride(num_coarserlayer + 1);
    Op[num_coarserlayer] = A;
    stride[num_coarserlayer] = num_planewaves_[start_layer];
    for(int i = num_coarserlayer - 1; i >= 0; --i) {
        int idx = L_ + i - num_coarserlayer;
        auto tmp = build_equation(idx);
        Op[i] = tmp.first.makeSparse();
        prolongation_op[i] = prolongation(idx);
        stride[i] = num_planewaves_[idx];
    }
    v_cycle(initial, eq_pair.second, Op, prolongation_op, stride, nu1, nu2, solve_coarest);
}

std::pair<HE_PUM::Vec_t, HE_PUM::Scalar> 
HE_PUM::power_multigird(size_type start_layer, int num_coarserlayer, 
        int mu1, int mu2) {
    LF_ASSERT_MSG((num_coarserlayer <= start_layer), 
        "please use a smaller number of wave layers");
    auto eq_pair = build_equation(start_layer);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
    std::vector<int> stride(num_coarserlayer + 1, 1);
    Op[num_coarserlayer] = A;
    for(int i = num_coarserlayer - 1; i >= 0; --i) {
        int idx = start_layer + i - num_coarserlayer;
        auto tmp = build_equation(idx);
        Op[i] = tmp.first.makeSparse();
        prolongation_op[i] = prolongation(idx);
    }

    int N = A.rows();
    /* Get the multigrid (2 grid) operator manually */
    // Op[0] = prolongation_op[0].transpose() * Op[1] * prolongation_op[0];
    // Mat_t mg_op = Mat_t::Identity(N, N) - 
    //     prolongation_op[0]*Op[0].colPivHouseholderQr().solve(prolongation_op[0].transpose())*Op[1];

    // Mat_t L = Mat_t(A.triangularView<Eigen::Lower>());
    // Mat_t U = L - A;
    // Mat_t GS_op = L.colPivHouseholderQr().solve(U);

    // Mat_t R_mu1 = Mat_t::Identity(N, N);
    // Mat_t R_mu2 = Mat_t::Identity(N, N);
    // for(int i = 0; i < mu1; ++i) {
    //     auto tmp = R_mu1 * GS_op;
    //     R_mu1 = tmp;
    // }
    // for(int i = 0; i < mu2; ++i) {
    //     auto tmp = R_mu2 * GS_op;
    //     R_mu2 = tmp;
    // }
    // auto tmp = R_mu2 * GS_op * R_mu1;
    // GS_op = tmp;

    // Vec_t eivals = mg_op.eigenvalues();

    // std::cout << eivals << std::endl;

    // Scalar domainant_eival = eivals(0);
    // for(int i = 1; i < eivals.size(); ++i) {
    //     if(std::abs(eivals(i)) > std::abs(domainant_eival)) {
    //         domainant_eival = eivals(i);
    //     }
    // }
    // std::cout << "Domainant eigenvalue: " << domainant_eival << std::endl;
    // std::cout << "Absolute value: " << std::abs(domainant_eival) << std::endl;
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
        if(r.norm() < 0.1) {
            break;
        }
        if(cnt > 20) {
            std::cout << "Power iteration for multigrid doesn't converge." << std::endl;
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue by power iteration: " << lambda << std::endl;
    return std::make_pair(u, lambda);
}