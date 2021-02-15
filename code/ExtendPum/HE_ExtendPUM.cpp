#include <vector>
#include <functional>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "HE_ExtendPUM.h"
#include "ExtendPUM_ElementMatrix.h"
#include "ExtendPUM_EdgeMat.h"
#include "ExtendPUM_EdgeVector.h"
#include "ExtendPUM_ElemVector.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

/*
 * Build equation A*x = phi in PUM spaces
 */
std::pair<lf::assemble::COOMatrix<HE_ExtendPUM::Scalar>, HE_ExtendPUM::Vec_t> 
HE_ExtendPUM::build_equation(size_type level) {
    auto mesh = mesh_hierarchy->getMesh(level);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    size_type N_wave(num_planwaves[level]);
    auto dofh = get_dofh(level);
    size_type N_dofs(dofh.NumDofs());

    // assemble for <grad(u), grad(v)> - k^2 <u,v>
    // (u, v) -> \int_K \alpha * (grad u, grad v) + \gamma * (u, v) dx
    ExtendPUM_ElementMatrix elmat_builder(N_wave, k, 1., -1. * k * k, degree);

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    // assemble boundary edge matrix, -i*k*u*v over \Gamma_R (outer boundary)
    // first need to distinguish between outer and inner boundar
    auto outer_boundary = outerBdy_selector(level);

    // (u,v) -> \int_e gamma * (u,v) dS
    ExtendPUM_EdgeMat edge_mat_builder(fe_space, outer_boundary, N_wave, k, -1i * k, degree);                                  
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector, \int_{\Gamma_R} gv.conj dS
    Vec_t phi(N_dofs);
    phi.setZero();
    // l(v) = \int_e g * v.conj dS(x)
    ExtendPUM_EdgeVec edgeVec_builder(fe_space, outer_boundary, N_wave, k, g, degree);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);

    if(hole_exist) {
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
      
        auto h_vec = fun_in_vec(level, h);
        // auto h_vec = h_in_vec(level, inner_boundary, inner_point);

        // auto h_vec_2 = h_in_vec(level, inner_boundary, inner_point);
        // std::cout << L2_BoundaryErr(level, h_vec, h, inner_boundary) << std::endl;
        // std::cout << L2_BoundaryErr(level, h_vec_2, h, inner_boundary) << std::endl;
        // if(level == 0){
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
                // vector representation of h
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
HE_ExtendPUM::Vec_t HE_ExtendPUM::fun_in_vec(size_type l, const FHandle_t& f) {
    auto mesh = mesh_hierarchy->getMesh(l);  // get mesh
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    size_type N_wave(num_planwaves[l]);
    auto dofh = get_dofh(l);
    size_type N_dofs(dofh.NumDofs());

    // assemble for \int u * v.conj dx
    ExtendPUM_ElementMatrix elmat_builder(N_wave, k, 0, 1., degree);

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);

    // assemble for \int (f,v) dx
    Vec_t phi(N_dofs);
    phi.setZero();
    ExtendPUM_ElemVec elvec_builder(fe_space, N_wave, k, f, degree);
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
HE_ExtendPUM::Vec_t HE_ExtendPUM::h_in_vec(size_type l, lf::mesh::utils::CodimMeshDataSet<bool> edge_selector,
    lf::mesh::utils::CodimMeshDataSet<bool> inner_point) {
    auto mesh = mesh_hierarchy->getMesh(l);  // get mesh
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    size_type N_wave(num_planwaves[l]);
    auto dofh = get_dofh(l);
    size_type N_dofs(dofh.NumDofs());

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);

    // assemble for \int_e (u,v) dx 
    ExtendPUM_EdgeMat edge_mat_builder(fe_space, edge_selector, N_wave, k, 1.0, degree);                                  
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);

    // assemble for \int_e (h,v) dx
    Vec_t phi(N_dofs);
    phi.setZero();
    ExtendPUM_EdgeVec edgeVec_builder(fe_space, edge_selector, N_wave, k, h, degree);
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
double HE_ExtendPUM::L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) {
    auto mesh = mesh_hierarchy->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

    size_type N_wave(num_planwaves[l]);
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
                for(int t = 0; t <= N_wave; ++t) {
                    Eigen::Vector2d d;
                    if(t == 0) {
                        d << 0.0, 0.0;
                    } else {
                        double pi = std::acos(-1);
                        d << std::cos(2*pi*(t-1)/N_wave), std::sin(2*pi*(t-1)/N_wave);
                    }
                    val_uh += mu(dofarray[i*(N_wave+1)+t]) * (X(0,i) + X(1,i)*x(0) + X(2,i)*x(1)) * std::exp(1i*k*d.dot(x));
                }
            }
            return std::abs((val_uh-val_u)*(val_uh-val_u));
        };
        res += std::abs(LocalIntegral(*cell, degree, integrand));
    }
    return std::sqrt(res);
}

double HE_ExtendPUM::H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) {
    auto mesh = mesh_hierarchy->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

    size_type N_wave(num_planwaves[l]);
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
                for(int t = 0; t <= N_wave; ++t) {
                    Eigen::Vector2d d, beta;
                    double pi = std::acos(-1);
                    if(t == 0) {
                        d << 0.0, 0.0;
                    } else {
                        d << std::cos(2*pi*(t-1)/N_wave), std::sin(2*pi*(t-1)/N_wave);
                    }
                    beta << X(1,i), X(2,i);
                    double lambda = X(0,i) + beta.dot(x);
                    val_grad_uh += mu(dofarray[i*(N_wave+1)+t]) * std::exp(1i*k*d.dot(x)) * (beta + 1i*k*lambda*d);
                }
            }
            return std::abs((val_grad_uh - val_grad_u).dot((val_grad_uh - val_grad_u)));
        };
        res += std::abs(LocalIntegral(*cell, degree, integrand));
    }
    return std::sqrt(res);
}

double HE_ExtendPUM::H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) {
    double l2err = L2_Err(l, mu, u);
    double h1serr = H1_semiErr(l, mu, grad_u);
    return std::sqrt(l2err*l2err + h1serr*h1serr);
}

// compute \int_{\partial Omega} |mu-u|^2dS
double HE_ExtendPUM::L2_BoundaryErr(size_type l, const Vec_t& mu, const FHandle_t& u,
    lf::mesh::utils::CodimMeshDataSet<bool> edge_selector) {
    auto mesh = mesh_hierarchy->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

    size_type N_wave(num_planwaves[l]);
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
                for(int t = 0; t <= N_wave; ++t) {
                    Eigen::Vector2d d;
                    double pi = std::acos(-1.);
                    if(t == 0){
                        d << 0.0, 0.0;
                    } else {
                        d << std::cos(2*pi*(t-1)/N_wave), std::sin(2*pi*(t-1)/N_wave);
                    }
                    val_uh += (mu(dofarray[t]) * tmp + mu(dofarray[t+N_wave+1]) * (1-tmp)) * std::exp(1i*k*d.dot(x));
                }
                return std::abs((val_uh-val_u)*(val_uh-val_u));
            };
            res += std::abs(LocalIntegral(*cell, degree, integrand));
        }
    }
    return std::sqrt(res);
}

HE_ExtendPUM::Vec_t HE_ExtendPUM::solve(size_type l) {
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
HE_ExtendPUM::SpMat_t HE_ExtendPUM::prolongation(size_type l) {
    LF_ASSERT_MSG((l < L), 
        "in prolongation, level should smaller than" << L);
    auto Q = Mat_t(prolongation_lagrange(l)); // n_{l+1} x n_l
    auto eq_pair = build_equation(l+1);
    auto A = eq_pair.first;
    Mat_t Phi = Assemble_EnergyProjection(l, 1.0, -k*k, -1i * k);
    Mat_t res(Phi.rows(), Phi.cols());
    int Nl = num_planwaves[l], Nl1 = num_planwaves[l+1];
    int n = Q.cols();
    
    for(int i = 0; i < n; ++i) {
        auto A_copy = A;
        auto selector = [&Q, &i, &Nl1](size_type dof_idx)->std::pair<bool, Scalar> {
            int j = dof_idx / (Nl1 + 1);
            if(Q(j, i) != 0) {
                return {false, 0};
            } else {
                return {true, 0};
            }
        };
        for(int t = 0; t <= Nl; ++t){
            Vec_t tmp_phi = Phi.col(i*(Nl+1)+t);
            lf::assemble::FixFlaggedSolutionCompAlt<Scalar>(selector, A_copy, tmp_phi);
            Phi.col(i*(Nl+1)+t) = tmp_phi;
        }
        SpMat_t A_crs = A_copy.makeSparse();
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(A_crs);
        res.block(0, i*(Nl+1), Phi.rows(), Nl+1) = solver.solve(Phi.block(0, i*(Nl+1), Phi.rows(), Nl+1));
    }
    return res.sparseView();
}

/*
 * assemble for 
 *  (z,c)-> \int_\Omega alpha*(grad z, grad c) + beta * (z,c)dx + gamma \int_\GammaR (z,c)dS 
 * 
 * and z \in W_l and c \in W_{l+1}
 * if we take z = b_i^l * e_s^l and c = b_j^{l+1} * e_t^{l+1}, it is not easy get (z, c)
 * so we make use of the fact that b_i^l = \sum qkj b_k^{l+1}
 * we assemble for z = b_i^{l+1} * e_s^l and c = b_j^{l+1} * e_t^{l+1}
 * 
 * To implement local energy projection, we can set
 *      alpha = 1.0, beta = -k^2, gamma = -ik
 * To implement local L2 projection, we can set
 *      alpha = 0.0, beta = 1.0, gamma = 1.0;
 */
HE_ExtendPUM::Mat_t HE_ExtendPUM::Assemble_EnergyProjection(size_type l, Scalar alpha, Scalar beta, Scalar gamma) {
    auto mesh = getmesh(l);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});
    auto Q = prolongation_lagrange(l); // n_{l+1} x n_l
    int Nl = num_planwaves[l], Nl1 = num_planwaves[l+1];
    int n = Q.rows();

    auto outer_boundary = outerBdy_selector(l);
    std::vector<triplet_t> triplets;
    // assemble for alpha*(grad z, grad c) + beta * (z,c)
    for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
    
        // Matrix storing corner coordinates in its columns(2x3 in this case)
        auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
        
        // suppose that the barycentric coordinate functions have the form
        // \lambda_i = X(0,i) + X(1,i)*x + X(2,i)*y
        // grad \lambda_i = [X(1,i), X(2,i)]^T
        Eigen::Matrix3d X, tmp;
        tmp.block<3,1>(0,0) = Eigen::Vector3d::Ones();
        tmp.block<3,2>(0,1) = vertices.transpose();
        X = tmp.inverse();

        // Number of shape functions covering current entity
        const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*cell));
        //obtain global indices of those shape functions
        nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};

        for(int i = 0; i < no_dofs; ++i) { 
            for(int s = 0; s <= Nl; ++s) {
                int z_idx = dofarray[i] * (Nl + 1) + s;
                for(int j = 0; j < no_dofs; ++j) { 
                    for(int t = 0; t <= Nl1; ++t) {
                        int v_idx = dofarray[j] * (Nl1 + 1) + t;

                        auto f = [this,&X,&i,&j,&s,&t,&Nl,&Nl1,&alpha,&beta](const Eigen::Vector2d& x)->Scalar {
                            Eigen::Vector2d ds, dt, betai, betaj; 
                            double pi = std::acos(-1);
                            if(s == 0) {
                                ds << 0.0, 0.0;
                            } else {
                                ds << std::cos(2*pi*(s-1)/Nl), std::sin(2*pi*(s-1)/Nl);
                            }
                            if(t == 0) {
                                dt << 0.0, 0.0;
                            } else {
                                dt << std::cos(2*pi*(t-1)/Nl1), std::sin(2*pi*(t-1)/Nl1);
                            }
                            betai << X(1, i), X(2, i);
                            betaj << X(1, j), X(2, j);
                            double lambdai = X(0,i) + betai.dot(x);
                            double lambdaj = X(0,j) + betaj.dot(x);

                            auto gradz = std::exp(1i*k*ds.dot(x)) * (betai + 1i*k*lambdai*ds);
                            auto gradv = std::exp(1i*k*dt.dot(x)) * (betaj + 1i*k*lambdaj*dt);
                            auto val_z = lambdai * std::exp(1i*k*ds.dot(x));
                            auto val_v = lambdaj * std::exp(1i*k*dt.dot(x));

                            return alpha * gradv.dot(gradz) + beta * val_z * std::conj(val_v);
                        };
                        Scalar tmp = LocalIntegral(*cell, degree, f);
                        triplets.push_back(triplet_t(v_idx, z_idx, tmp));
                    }
                }
            }
        }
    }
    // assemble for gamma * (z,c)dS 
    for(const lf::mesh::Entity* cell: mesh->Entities(1)) {
        if(!outer_boundary(*cell)){
            continue;
        }
        const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        // Matrix storing corner coordinates in its columns(2x2 in this case)
        auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
        const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*cell));
        //obtain global indices of those shape functions
        nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};

        for(int i = 0; i < no_dofs; ++i) { 
            for(int s = 0; s <= Nl; ++s) {
                int z_idx = dofarray[i] * (Nl + 1) + s;
                for(int j = 0; j < no_dofs; ++j) { 
                    for(int t = 0; t <= Nl1; ++t) {
                        int v_idx = dofarray[j] * (Nl1 + 1) + t;
                        auto integrand = [this,&vertices,&i,&j,&s,&t,&Nl,&Nl1,&gamma](const Eigen::Vector2d& x)->Scalar {
                            double x1, y1, x2, y2, tmp;
                            x1 = vertices(0,0), y1 = vertices(1,0);
                            x2 = vertices(0,1), y2 = vertices(1,1);
                            if(x1 == x2) {
                                tmp = (x(1) - y2) / (y1 - y2);
                            } else {
                                tmp = (x(0) - x2) / (x1 - x2); 
                            }
                            Scalar lambdai = (i == 0 ? tmp : 1 - tmp);
                            Scalar lambdaj = (j == 0 ? tmp : 1 - tmp);

                            if(s == 0) {
                                ds << 0.0, 0.0;
                            } else {
                                ds << std::cos(2*pi*(s-1)/Nl), std::sin(2*pi*(s-1)/Nl);
                            }
                            if(t == 0) {
                                dt << 0.0, 0.0;
                            } else {
                                dt << std::cos(2*pi*(t-1)/Nl1), std::sin(2*pi*(t-1)/Nl1);
                            }
                            Scalar val_z = lambdai * std::exp(1i*k*ds.dot(x));
                            Scalar val_v = lambdaj * std::exp(1i*k*dt.dot(x));
                            return val_z * std::conj(val_v);
                        };
                        Scalar tmp = LocalIntegral(*cell, degree, f);
                        triplets.push_back(triplet_t(v_idx, z_idx, tmp));
                    }
                }
            }
        }
    }

    SpMat_t Z(n*Nl1, n*Nl), ;
    Z.setFromTriples(triplets.begin(), triplets.end());
    
    Mat_t Phi = Mat_t::Zero(n*Nl1, Q.cols()*Nl);
    Mat_t Dense_Q = Mat_t(Q);
    Mat_t Dense_Z = Mat_t(Z);
    for(int J = 0; J < Q.cols()*(Nl+1); ++J) {
        int j = J / (Nl + 1);
        int s = J % (Nl + 1);
        for(int I = 0; I < n*(Nl1+1); ++I) {
            int i = I / (Nl1 + 1);
            int t = I % (Nl1 + 1);
            for(int k = 0; k < n; ++k) {
                Phi(I,J) += Dense_Q(k,j) * Dense_Z(I, k*(Nl+1) + s);
            }
        }
    }
    return Phi;
}

HE_ExtendPUM::SpMat_t HE_ExtendPUM::prolongation_SE_S() {
    double pi = std::acos(-1.);
    auto Q = prolongation_lagrange(L-1);
    int n = Q.rows();
    int N = num_planwaves[L-1];

    Mat_t E = Mat_t::Zero(n, N+1);

    auto S_mesh = getmesh(L);
    auto S_dofh = lf::assemble::UniformFEDofHandler(S_mesh, 
                        {{lf::base::RefEl::kPoint(), 1}});
    for(int i = 0; i < n; ++i) {
        const lf::mesh::Entity& vi = S_dofh.Entity(i); // the entity to which i-th global shape function is associated
        coordinate_t vi_coordinate = lf::geometry::Corners(*vi.Geometry()).col(0);
        E(i,0) = 1.0;
        for(int t = 1; t <= N; ++t) {
            Eigen::Vector2d dt;
            dt << std::cos(2*pi*(t-1)/N), std::cos(2*pi*(t-1)/N);
            E(i,t) = std::exp(1i*k*dt.dot(vi_coordinate));
        }
    }

    SpMat_t res(n, Q.cols() * (N+1));
    std::vector<triplet_t> triplets;
    for(int outer_idx = 0; outer_idx < Q.outerSize(); ++outer_idx) {
        for(SpMat_t::InnerIterator it(Q, outer_idx); it; ++it) {
            int i = it.row();
            int j = it.col();
            Scalar qij = it.value(); 
            for(int k = 0; k <= N; ++k) {
                triplets.push_back(triplet_t(i, j*(N+1)+k, qij*E(i,k)));
            }
        }
    }
    res.setFromTriplets(triplets.begin(), triplets.end());
    return res;
}

HE_ExtendPUM::Vec_t HE_ExtendPUM::solve_multigrid(size_type start_layer, int num_coarserlayer, 
    int mu1, int mu2) {

    LF_ASSERT_MSG((num_coarserlayer <= L), 
        "please use a smaller number of wave layers");
    auto eq_pair = build_equation(L);
    SpMat_t A(eq_pair.first.makeSparse());

    std::vector<SpMat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
    std::vector<int> stride(num_coarserlayer + 1);
    Op[num_coarserlayer] = A;
    stride[num_coarserlayer] = num_planwaves[start_layer];
    for(int i = num_coarserlayer - 1; i >= 0; --i) {
        int idx = L + i - num_coarserlayer;
        auto tmp = build_equation(idx);
        Op[i] = tmp.first.makeSparse();
        prolongation_op[i] = prolongation(idx);
        stride[i] = num_planwaves[idx];
    }
    Vec_t initial = Vec_t::Random(A.rows());
    v_cycle(initial, eq_pair.second, Op, prolongation_op, stride, mu1, mu2);
    return initial;
}

std::pair<HE_ExtendPUM::Vec_t, HE_ExtendPUM::Scalar> 
HE_ExtendPUM::power_multigird(size_type start_layer, int num_coarserlayer, 
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