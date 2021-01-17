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

using namespace std::complex_literals;

std::pair<lf::assemble::COOMatrix<HE_LagrangeO1::Scalar>, HE_LagrangeO1::Vec_t>
HE_LagrangeO1::build_equation(size_type level) {
    
    auto mesh = mesh_hierarchy->getMesh(level);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    // auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});
    auto dofh = get_dofh(level);
    
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
                for(int i = level; i > 0; --i) {
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
    	edgeVec_builder(fe_space, mf_g, outer_boundary);
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

double HE_LagrangeO1::L2_norm(size_type l, const Vec_t& mu) {
    // auto mesh = getmesh(l);
    // auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    // // create mesh function represented by coefficient vector
    // auto mf_mu = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu);
    // auto mf_mu_conj = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu.conjugate());

    // double res = std::abs(lf::uscalfe::IntegrateMeshFunction(*mesh, mf_mu * mf_mu_conj, 10));
    // return std::sqrt(res);
    auto dofh = get_dofh(l);

    int N_dofs = dofh.NumDofs();
    lf::assemble::COOMatrix<double> mass_matrix(N_dofs, N_dofs);
    
    lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
    lf::mesh::utils::MeshFunctionConstant<double> mf_zero(0);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(getmesh(l));
    // (u,v)->(grad u, grad v) + (u,v)
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_identity)>
        elmat_builder(fe_space, mf_zero, mf_identity);
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mass_matrix);
    
    const Eigen::SparseMatrix<double> mass_mat = mass_matrix.makeSparse();
    double res = std::abs(mu.dot(mass_mat * mu.conjugate()));
    return std::sqrt(res);
}

// return ||u-mu||_2, mu is represented by vector
// double L2Err_norm(std::shared_ptr<lf::mesh::Mesh> mesh, const function_type& u, const vec_t& mu) {
//     auto mesh = mesh_hierarchy->getMesh(l);
//     auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
//     // u has to be wrapped into a mesh function for error computation
//     lf::mesh::utils::MeshFunctionGlobal mf_u{u};
//     // create mesh function representing solution 
//     auto mf_mu = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu);

//     // conjugate functions
//     auto u_conj = [&u](const Eigen::Vector2d& x) -> Scalar {
//         return std::conj(u(x));};
//     lf::mesh::utils::MeshFunctionGlobal mf_u_conj{u_conj};
//     auto mf_mu_conj = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu.conjugate());
    
//     auto mf_square = (mf_u - mf_mu) * (mf_u_conj - mf_mu_conj);
//     double L2err = std::abs(lf::uscalfe::IntegrateMeshFunction(*mesh, mf_square, 10));
//     return std::sqrt(L2err);
// }

double HE_LagrangeO1::H1_norm(size_type l, const Vec_t& mu) {
    auto dofh = get_dofh(l);

    int N_dofs = dofh.NumDofs();
    lf::assemble::COOMatrix<double> mass_matrix(N_dofs, N_dofs);
    
    lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(getmesh(l));
    // (u,v)->(grad u, grad v) + (u,v)
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_identity)>
        elmat_builder(fe_space, mf_identity, mf_identity);
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mass_matrix);
    
    const Eigen::SparseMatrix<double> mass_mat = mass_matrix.makeSparse();
    double res = std::abs(mu.dot(mass_mat * mu.conjugate()));
    return std::sqrt(res);
}

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