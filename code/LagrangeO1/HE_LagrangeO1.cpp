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

#include "HE_LagrangeO1.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

std::pair<lf::assemble::COOMatrix<HE_LagrangeO1::Scalar>, HE_LagrangeO1::Vec_t>
HE_LagrangeO1::build_equation(size_type l) {
    
    auto mesh = mesh_hierarchy->getMesh(l);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    auto dofh = get_dofh(l);
    
    // assemble for <grad(u), grad(v)> - k^2 uv
    lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
    lf::mesh::utils::MeshFunctionConstant<double> mf_k(-1. * k * k);
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_k)> 
    	elmat_builder(fe_space, mf_identity, mf_k);
    
    size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    // assemble boundary edge matrix, -i*k*u*v over \Gamma_R (outer boundary)
    lf::mesh::utils::MeshFunctionConstant<Scalar> mf_ik(-1i * k);

    auto outer_boundary{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)}; 

    if(hole_exist){
        // there is a hole inside, need to distinguish between outer and inner boundar
        auto outer_nr = reader->PhysicalEntityName2Nr("outer_boundary");
        auto inner_nr = reader->PhysicalEntityName2Nr("inner_boundary");

        // modify it to classify inner and outer boundary
        for(const lf::mesh::Entity* edge: mesh->Entities(1)) {
            if(outer_boundary(*edge)) {
                // find a boundary edge, need to determine if it's outer boundary
                const lf::mesh::Entity* parent_edge = edge;
                for(int i = l; i > 0; --i) {
                    parent_edge = mesh_hierarchy->ParentEntity(i, *parent_edge);
                }
                if(reader->IsPhysicalEntity(*parent_edge, inner_nr)) {
                    // it is the inner boundary
                    outer_boundary(*edge) = false;
                }
            }
        }
    }
                                             
    lf::uscalfe::MassEdgeMatrixProvider<double, decltype(mf_ik), decltype(outer_boundary)>
    	edge_mat_builder(fe_space, mf_ik, outer_boundary);
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector, \int_{\Gamma_R} gvdS
    Vec_t phi(N_dofs);
    phi.setZero();
    lf::mesh::utils::MeshFunctionGlobal mf_g{g};
    lf::mesh::utils::MeshFunctionGlobal mf_h{h};
    lf::uscalfe::ScalarLoadEdgeVectorProvider<double, decltype(mf_g), decltype(outer_boundary)>
    	edgeVec_builder(fe_space, mf_g, 
        lf::quad::make_QuadRule(lf::base::RefEl::kSegment(), 10), outer_boundary);
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
        std::vector<std::pair<bool, Scalar>> ess_dof_select{};
        for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
            const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
            const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
            const Scalar h_val = h(node_pos);
            if(inner_point(dof_node)) {
                // Dof associated with a entity on the boundary: "essential dof"
                // The value of the dof should be set to the value of the function h
                // at the location of the node.
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

HE_LagrangeO1::SpMat_t HE_LagrangeO1::prolongation(size_type l) {
    return prolongation_lagrange(l);
}

HE_LagrangeO1::Vec_t HE_LagrangeO1::solve(size_type l) {
    auto eq_pair = build_equation(l);
    const SpMat_t A_crs(eq_pair.first.makeSparse());
    Eigen::SparseLU<SpMat_t> solver;
    solver.compute(A_crs);
    Vec_t fe_sol;
    if(solver.info() == Eigen::Success) {
        fe_sol = solver.solve(eq_pair.second);
    } else {
        LF_ASSERT_MSG(false, "Eigen Factorization failed");
    }
    return fe_sol;
}

HE_LagrangeO1::Vec_t HE_LagrangeO1::solve_multigrid(size_type start_layer, int num_coarserlayer, 
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
        prolongation_op[i] = prolongation(idx);
        Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
        // auto tmp = build_equation(idx);
        // Op[i] = tmp.first.makeSparse();
    }

    Vec_t initial = Vec_t::Random(A.rows());
    v_cycle(initial, eq_pair.second, Op, prolongation_op, stride, mu1, mu2);
    return initial;
}

std::pair<HE_LagrangeO1::Vec_t, HE_LagrangeO1::Scalar> 
HE_LagrangeO1::power_multigird(size_type start_layer, int num_coarserlayer, int mu1, int mu2) {
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
    if(num_coarserlayer == 1) {
        Mat_t mg_op = Mat_t::Identity(N, N) - 
            prolongation_op[0]*Mat_t(Op[0]).colPivHouseholderQr().solve(Mat_t(prolongation_op[0]).transpose())*Op[1];

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

        std::cout << eivals << std::endl;

        Scalar domainant_eival = eivals(0);
        for(int i = 1; i < eivals.size(); ++i) {
            if(std::abs(eivals(i)) > std::abs(domainant_eival)) {
                domainant_eival = eivals(i);
            }
        }
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

double HE_LagrangeO1::L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u){
    auto mesh = mesh_hierarchy->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    // u has to be wrapped into a mesh function for error computation
    lf::mesh::utils::MeshFunctionGlobal mf_u{u};
    // create mesh function representing finite element solution 
    auto mf_mu = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu);

    // conjugate functions
    auto u_conj = [&u](const Eigen::Vector2d& x) -> Scalar {
        return std::conj(u(x));};
    lf::mesh::utils::MeshFunctionGlobal mf_u_conj{u_conj};
    auto mf_mu_conj = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu.conjugate());
    
    auto mf_square = (mf_u - mf_mu) * (mf_u_conj - mf_mu_conj);
    double L2err = std::abs(lf::uscalfe::IntegrateMeshFunction(*mesh, mf_square, 10));
    return std::sqrt(L2err);
}

double HE_LagrangeO1::H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) {
    auto mesh = mesh_hierarchy->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

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
        // then uh(FE solution) restricted in cell is \sum_i \lambda_i * mu(dofarray(i))

        auto grad_uh = (Eigen::Matrix<Scalar,2,1>() << 0.0, 0.0).finished();
        for(size_type i = 0; i < no_dofs; ++i) {
            grad_uh += mu(dofarray[i]) * X.block<2,1>(1,i);
        }
        // construct ||grad uh - grad u||^2_{cell}
        auto integrand = [&grad_uh, &grad_u](const Eigen::Vector2d& x)->Scalar {
            return std::abs((grad_uh - grad_u(x)).dot(grad_uh - grad_u(x)));
        };
        res += std::abs(LocalIntegral(*cell, 20, integrand));
    }
    return std::sqrt(res);
}

double HE_LagrangeO1::H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) {
    double l2err = L2_Err(l, mu, u);
    double h1serr = H1_semiErr(l, mu, grad_u);
    return std::sqrt(l2err * l2err + h1serr * h1serr);
}

// Nodal projection
HE_LagrangeO1::Vec_t HE_LagrangeO1::fun_in_vec(size_type l, const FHandle_t& f) {
    auto dofh = get_dofh(l);
    size_type N_dofs(dofh.NumDofs());
    Vec_t res(N_dofs);
    for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
        const lf::mesh::Entity& dof_node{dofh.Entity(dofnum)};
        const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
        res(dofnum) = f(node_pos);
    }
    return res;
}