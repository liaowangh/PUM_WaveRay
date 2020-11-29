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
#inlucde "pum_fem.h"

using namespace std::complex_literals;

lf::assemble::UniformFEDofHandler PUM_FEM::generate_dof(size_type level) {
    auto mesh = mesh_hierarchy->getMesh(level);
    size_type num = level == L ? 1 : std::pow(2, L + 1 - level);
    return lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), num}});
}

void PUM_FEM::generate_dof::Prolongation_P(){
    P = std::vector<elem_mat_t>(L-1);
    for(int l = 0; l < L - 1; ++l) {
        auto coarse_mesh = mesh_hierarchy->getMesh(l);
        auto fine_mesh = mesh_hierarchy->getMesh(l+1);
        
        auto coarse_dofh = lf::assemble::UniformFEDofHandler(coarse_mesh,{{lf::base::RefEl::kPoint(), 1}});
        auto fine_dof = lf::assemble::UniformFEDofHandler(fine_mesh,{{lf::base::RefEl::kPoint(), 1}});
        
        // TODO: how to handle when the boundary is not included
        size_type n_c = coarse_dofh.NumDofs();
        size_type n_f = fine_dof.NumDofs();
        
        elem_mat_t M(n_c, n_f);
        
        for(auto &edge: fine_mesh->Entities(1)) {
            auto points{edge->SubEntities(1)};
            size_type num_points = edge->NumSubEntities(1);
            for(int j = 0; j < num_points; ++j) {
                // TODO: check whether the coarser grid can recognize a verticle from the finer grid
                if(coarse_mesh->contains(points[j])){
                    M(coarse_mesh->Index(points[j]), coarse_mesh->Index(points[j])) = 1;
                
                    M(coarse_mesh->Index(points[j]),coarse_mesh->Index(points[1-j])) = 0.5;
                }
            }
        }
        P[l] = M.transpose();
    }
}


PUM_FEM::mat_scalar PUM_FEM::int_mesh(int level, lf::uscalfe::MeshFunctionGlobal f) {
    mat_scalar res = 0;
    
    // traverse all triangles
    for(auto e: mesh_hierarchy->getMesh(level)->Entities(0)){
        Eigen::MatrixXd corners = lf::geometry::Corners(*(e->Geometry()));
        double area = lf::geometry::Volume(*(e->Geometry()));
        mat_scalar tmp = 0;
        res += (f(corners.col(0)) + f(corners.col(1)) + f(corners.col(2))) * area / 3'
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
    Q = std::vector<elem_mat_t>(L-1);
    
    lf::uscalfe::MeshFunctionGlobal mf_one{[](Eigen::Vector2d x)->double{return 1.0;}};
    
    for(int l = 0; l < L - 1; ++l) {
        int N1 = std::pow(2, L + 1 - l);
        int N2 = std::pow(2, L - l);
        
        elem_mat_t M(N2, N1);
        for(int i = 0; i < N2; ++i) {
            
            auto dl1t = generate_fre(L, k, l, i);
            auto dl1t1 = generate_fre(L, k, l, i+1);
            auto dlt = generate_fre(L, k, l, 2*i+1);
            
            
            Eigen::Matrix<mat_scalar, 2, 2> A;
            A << int_mesh(L, mf_one),
                int_mesh(L, exp_wave(dl1t1[0] - dl1t[0], dl1t1[1] - dl1t[1])),
                int_mesh(L, exp_wave(dl1t[0] - dl1t1[0], dl1t[1] - dl1t1[1])),
                int_mesh(L, mf_one);
            Eigen::Matrix<mat_scalar, 2, 1> b;
            b << int_mesh(L, exp_wave(dlt[0] - dl1t[0], dlt[1] - dl1t[1])),
                int_mesh(L, exp_wave(dlt[0] - dl1t1[0], dlt[1] - dl1t1[1]));
            auto tmp = A.colPivHouseholderQr().solve(b);
            
            M(i, 2 * i) = 1;
            M(i, 2 * i + 1) = tmp(0);
            M((i+1) / N2, 2 * i + 1) = tmp(1);
        }
        Q[l] = M;
    }
}

std::pair<PUM_FEM::elem_mat_t, PUM_FEM::res_vec_t>
PUM_FEM::build_finest() {
    
    auto mesh = mesh_hierarchy->getMesh(L);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    auto dofh = lf::assemble::UniformFEDofHandler(mesh,{{lf::base::RefEl::kPoint(), 1}});
    
    lf::uscalfe::MeshFunctionGlobal mf_identity{1.};
    lf::uscalfe::MeshFunctionGlobal mf_k{-k * k};
    lf::uscalfe::MeshFunctionGlobal mf_ik{-1i*k};
    
    lf::uscalfe::ReactionDiffusionElementProvider<mat_scalar, decltype(mf_identity), decltype(mf_k)> elmat_builder(fe_space, mf_identity, mf_k);
    
    size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<mat_scalar> A(dofh.Numdofs(), dofh.NumDofs());
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    // TODO: I need to modify the edge selector
    // assemble boundary edge matrix
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(dofh.Mesh(), 1)};
                                                            
    auto my_selector = [&dofh, &bd_flags](unsigned int dof_idx){
        if (bd_flags(dof_handler.Entity(dof_idx))) {
            return (std::pair<bool, double>(true, 1));
        }else{
            return (std::pair<bool, double>(false, 0));
        }
    };
                                                        
    lf::usclafe::MassEdgeMatrixProvider<mat_scalar, decltype(mf_ik), decltype(my_selector)> edge_mat_builder(fe_space, mf_ik, my_selector);
    lf::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector
    rhs_vec_t phi(N_dofs);
    phi.setZero();
    lf::uscalfe::ScalarLoadEdgeVectorProvider<mat_scalar, decltype(g), decltype(my_selector)> edgeVec_builder(fe_sapce, g, my_selector);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder);
    
    return {A, phi};
}
