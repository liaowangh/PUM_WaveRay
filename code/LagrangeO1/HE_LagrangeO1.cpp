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

HE_LagrangeO1::Mat_t HE_LagrangeO1::prolongation(size_type l) {
    // P_Lagrange = std::vector<Mat_t>(L);
    LF_ASSERT_MSG(l >= 0 && l < L, "l in prolongation should be smaller to L");
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
    return M.transpose();
}

HE_LagrangeO1::Vec_t HE_LagrangeO1::solve(size_type l) {
    auto eq_pair = build_equation(l);
    const Eigen::SparseMatrix<Scalar> A_crs(eq_pair.first.makeSparse());
    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A_crs);
    Vec_t fe_sol;
    if(solver.info() == Eigen::Success) {
        fe_sol = solver.solve(eq_pair.second);
    } else {
        LF_ASSERT_MSG(false, "Eigen Factorization failed")
    }
    return fe_sol;
}

void HE_LagrangeO1::solve_multigrid(size_type start_layer, Vec_t& initial, int num_coarserlayer, 
    int mu1, int mu2) {

    LF_ASSERT_MSG((num_coarserlayer <= start_layer), 
        "please use a smaller number of wave layers");
    auto eq_pair = build_equation(start_layer);
    Mat_t A(eq_pair.first.makeDense());

    std::vector<Mat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
    std::vector<int> stride(num_coarserlayer + 1, 1);
    Op[num_coarserlayer] = A;
    for(int i = num_coarserlayer - 1; i >= 0; --i) {
        int idx = start_layer + i - num_coarserlayer;
        std::cout << idx << std::endl;
        auto tmp = build_equation(idx);
        Op[i] = tmp.first.makeDense();
        // prolongation_op[i] = P_Lagrange[idx];
        prolongation_op[i] = prolongation(idx);
    }

    /* debugging */
    // std::cout << "operator size" << std::endl;
    // for(int i = 0; i < Op.size(); ++i){
    //     std::cout << i << " " << Op[i].rows() << std::endl;
    // }

    // std::cout << "transfer operator size" << std::endl;
    // for(int i = 0; i < prolongation_op.size(); ++i){
    //     std::cout << i << " [" << prolongation_op[i].rows() << "," 
    //                            << prolongation_op[i].cols() << "]" << std::endl;
    // }
    // std::cout << "initial size" << std::endl;
    // std::cout << initial.size() << std::endl;
    /* debugging */
    v_cycle(initial, eq_pair.second, Op, prolongation_op, stride, mu1, mu2);
}

HE_LagrangeO1::Vec_t HE_LagrangeO1::power_multigird(size_type start_layer, int num_coarserlayer, 
    int mu1, int mu2) {
    LF_ASSERT_MSG((num_coarserlayer <= start_layer), 
        "please use a smaller number of wave layers");
    auto eq_pair = build_equation(start_layer);
    Mat_t A(eq_pair.first.makeDense());

    std::vector<Mat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
    std::vector<int> stride(num_coarserlayer + 1, 1);
    Op[num_coarserlayer] = A;
    for(int i = num_coarserlayer - 1; i >= 0; --i) {
        int idx = start_layer + i - num_coarserlayer;
        auto tmp = build_equation(idx);
        Op[i] = tmp.first.makeDense();
        // prolongation_op[i] = P_Lagrange[idx];
        prolongation_op[i] = prolongation(idx);
    }

    int N = A.rows();
    Vec_t initial = Vec_t::Random(N);
    initial.normalize();
    Vec_t old_vec;
    Vec_t zero_vec = Vec_t::Zero(N);
    Scalar lambda;
    int cnt = 0;
    
    std::cout << std::left << std::setw(10) << "Iteration" 
        << std::setw(20) << "residual_norm" << std::endl;
    while(true) {
        cnt++;
        old_vec = initial;
        v_cycle(initial, zero_vec, Op, prolongation_op, stride, mu1, mu2);
        
        lambda = old_vec.dot(initial);  // domainant eigenvalue
        auto r = initial - lambda * old_vec;
        
        initial.normalize();
    
        if(cnt % 10 == 0) {
            std::cout << std::left << std::setw(10) << cnt 
                << std::setw(20) << r.norm() 
                << std::setw(20) << (initial - old_vec).norm()
                << std::endl;
        }
        if(r.norm() < 0.1) {
            break;
        }
        if(cnt > 50) {
            std::cout << "Power iteration for multigrid doesn't converge." << std::endl;
            break;
        }
    }
    std::cout << "Number of iterations: " << cnt << std::endl;
    std::cout << "Domainant eigenvalue by power iteration: " << lambda << std::endl;
    return initial;
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
        res += std::abs(LocalIntegral(*cell, 10, integrand));
    }
    return std::sqrt(res);
}

double HE_LagrangeO1::H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) {
    double l2err = L2_Err(l, mu, u);
    double h1serr = H1_semiErr(l, mu, grad_u);
    return std::sqrt(l2err * l2err + h1serr * h1serr);
}

// Nodal projection, may not be the best ideal method.
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