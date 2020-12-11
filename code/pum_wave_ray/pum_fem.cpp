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

#include "EdgeMat.h"
#include "ElementMatrix.h"
#include "pum_fem.h"

using namespace std::complex_literals;

lf::assemble::UniformFEDofHandler PUM_FEM::generate_dof(unsigned int level) {
    auto mesh = mesh_hierarchy->getMesh(level);
    size_type num = level == L_ ? 1 : std::pow(2, L_ + 1 - level);
    return lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), num}});
}

PUM_FEM::PUM_FEM(unsigned int L, double k, std::string mesh_path, PUM_FEM::function_type g, PUM_FEM::function_type h): 
    L_(L), k_(k), g_(g), h_(h) {
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    reader = std::make_shared<lf::io::GmshReader>(std::move(mesh_factory), mesh_path);
    mesh_hierarchy = lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(reader->mesh(), L);
}

void PUM_FEM::Prolongation_P(){
    P = std::vector<elem_mat_t>(L_);
    for(int l = 0; l < L_; ++l) {
        auto coarse_mesh = mesh_hierarchy->getMesh(l);
        auto fine_mesh = mesh_hierarchy->getMesh(l+1);
        
        auto coarse_dofh = lf::assemble::UniformFEDofHandler(coarse_mesh,{{lf::base::RefEl::kPoint(), 1}});
        auto fine_dof = lf::assemble::UniformFEDofHandler(fine_mesh,{{lf::base::RefEl::kPoint(), 1}});
        
        size_type n_c = coarse_dofh.NumDofs();
        size_type n_f = fine_dof.NumDofs();
        
        elem_mat_t M(n_c, n_f);
        
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
        P[l] = M.transpose();
    }
}



PUM_FEM::mat_scalar PUM_FEM::integration_mesh(int level, PUM_FEM::function_type f) {
    mat_scalar res = 0;
    
    // traverse all triangles
    for(auto e: mesh_hierarchy->getMesh(level)->Entities(0)){
        Eigen::MatrixXd corners = lf::geometry::Corners(*(e->Geometry()));
        double area = lf::geometry::Volume(*(e->Geometry()));
        mat_scalar tmp = 0;
        res += (f(corners.col(0)) + f(corners.col(1)) + f(corners.col(2))) * area / 3.;
    }
    return res;
}


std::vector<double> generate_fre(int L, double k, int l, int t) {
    int N = (1 << (L + 1 - l));
    double d1 = std::cos(2*M_PI*t / N);
    double d2 = std::cos(2*M_PI*t / N);
    return {d1, d2};
}

void PUM_FEM::Prolongation_Q() {
    Q = std::vector<elem_mat_t>(L_);
    
    auto identity = [](Eigen::Vector2d x)->mat_scalar{ return 1.0;};
    
    for(int l = 0; l < L_; ++l) {
        int N1 = std::pow(2, L_ + 1 - l);
        int N2 = std::pow(2, L_ - l);
        
        elem_mat_t M(N2, N1);
        for(int i = 0; i < N2; ++i) {
            
            auto dl1t = generate_fre(L_, k_, l, i);
            auto dl1t1 = generate_fre(L_, k_, l, i+1);
            auto dlt = generate_fre(L_, k_, l, 2*i+1);
            
            Eigen::Matrix<mat_scalar, 2, 2> A;
            A << integration_mesh(L_, identity),
                integration_mesh(L_, exp_wave(dl1t1[0] - dl1t[0], dl1t1[1] - dl1t[1])),
                integration_mesh(L_, exp_wave(dl1t[0] - dl1t1[0], dl1t[1] - dl1t1[1])),
                integration_mesh(L_, identity);
            Eigen::Matrix<mat_scalar, 2, 1> b;
            b << integration_mesh(L_, exp_wave(dlt[0] - dl1t[0], dlt[1] - dl1t[1])),
                integration_mesh(L_, exp_wave(dlt[0] - dl1t1[0], dlt[1] - dl1t1[1]));
            auto tmp = A.colPivHouseholderQr().solve(b);
            
            M(i, 2 * i) = 1;
            M(i, 2 * i + 1) = tmp(0);
            M((i+1) / N2, 2 * i + 1) = tmp(1);
        }
        Q[l] = M;
    }
}


/*
std::pair<lf::assemble::COOMatrix<PUM_FEM::mat_scalar>, PUM_FEM::rhs_vec_t>
PUM_FEM::build_equation(size_type level) {
    
    auto mesh = mesh_hierarchy->getMesh(level);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<mat_scalar>>(mesh);
    
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});
    
    // assemble for <grad(u), grad(v)> - k^2 uv
    auto identity = [this](Eigen::Vector2d x) -> mat_scalar { return 1.; };
    lf::mesh::utils::MeshFunctionGlobal mf_identity{identity};
    auto f_k = [this](Eigen::Vector2d x) -> mat_scalar { return -1 * k_ * k_;}; 
    lf::mesh::utils::MeshFunctionGlobal mf_k{f_k};
    // lf::uscalfe::ReactionDiffusionElementMatrixProvider<mat_scalar, decltype(mf_identity), decltype(mf_k)> 
    // 	elmat_builder(fe_space, mf_identity, mf_k);
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<mat_scalar, decltype(mf_identity), decltype(mf_k)> 
    	elmat_builder(fe_space, mf_identity, mf_k);
    
    size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<mat_scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    // assemble boundary edge matrix, -i*k*u*v over \Gamma_R (outer boundary)
    auto f_ik = [this](Eigen::Vector2d x) -> mat_scalar { return -1i * k_;};
    lf::mesh::utils::MeshFunctionGlobal mf_ik{f_ik};

    // first need to distinguish between outer and inner boundar
    auto outer_nr = reader->PhysicalEntityName2Nr("outer_boundary");
    auto inner_nr = reader->PhysicalEntityName2Nr("inner_boundary");

    auto outer_boundary{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)}; 
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
                                                   
    lf::uscalfe::MassEdgeMatrixProvider<mat_scalar, decltype(mf_ik), decltype(outer_boundary)> 
    	edge_mat_builder(fe_space, mf_ik, outer_boundary);
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector, \int_{\Gamma_R} gvdS
    rhs_vec_t phi(N_dofs);
    phi.setZero();
    lf::mesh::utils::MeshFunctionGlobal mf_g{g_};
    lf::mesh::utils::MeshFunctionGlobal mf_h{h_};
    lf::uscalfe::ScalarLoadEdgeVectorProvider<mat_scalar, decltype(mf_g), decltype(outer_boundary)> 
    	edgeVec_builder(fe_space, mf_g, outer_boundary);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);
    

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
    std::vector<std::pair<bool, mat_scalar>> ess_dof_select{};
    for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
        const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
        const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
        const mat_scalar h_val = h_(node_pos);
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
    lf::assemble::FixFlaggedSolutionCompAlt<mat_scalar>(
        [&ess_dof_select](size_type dof_idx)->std::pair<bool, mat_scalar> {
            return ess_dof_select[dof_idx];},
    A, phi);
    return std::make_pair(A, phi);
}
*/