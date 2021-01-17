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

using namespace std::complex_literals;

/*
 * Build equation A*x = phi in PUM spaces
 */
std::pair<lf::assemble::COOMatrix<HE_PUM::Scalar>, HE_PUM::Vec_t> 
HE_PUM::build_equation(size_type level) {
    auto mesh = mesh_hierarchy->getMesh(level);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    size_type N_wave(num_planwaves[level]);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), N_wave}});
    size_type N_dofs(dofh.NumDofs());

    // assemble for <grad(u), grad(v)> - k^2 <u,v>
    // (u, v) -> \int_K \alpha grad u grad v.conj + \gamma * u v.conj dx
    PUM_FEElementMatrix elmat_builder(N_wave, k, 1., -1. * k * k);

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    auto outer_boundary{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)};
    if(hole_exist) {
        // assemble boundary edge matrix, -i*k*u*v over \Gamma_R (outer boundary)
        // first need to distinguish between outer and inner boundar
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

    // (u,v) -> \int_e gamma * u * v.conj dS
    PUM_EdgeMat edge_mat_builder(fe_space, outer_boundary, N_wave, k, -1i * k);                                  
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector, \int_{\Gamma_R} gvdS
    Vec_t phi(N_dofs);
    phi.setZero();
    // l(v) = \int_e g * v.conj dS(x)
    PUM_EdgeVec edgeVec_builder(fe_space, outer_boundary, N_wave, k, g);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);

    return std::make_pair(A, phi);
}

/*
 * To get the vector representation of function f w.r.t the PUM space,
 * a L2 projection is used.
 * P2(f) = uh, such that ||f-uh|| is minimized, which is equivalent to 
 * find uh such that (uh, v) = (f,v) for all v in the PUM space
 */
HE_PUM::Vec_t HE_PUM::fun_in_vec(size_type l, const FHandle_t& f) {
    auto mesh = mesh_hierarchy->getMesh(l);  // get mesh
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    size_type N_wave(num_planwaves[l]);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), N_wave}});
    size_type N_dofs(dofh.NumDofs());

    // assemble for \int u * v.conj dx
    PUM_FEElementMatrix elmat_builder(N_wave, k, 0, 1.);

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);

    // assemble for \int (f,v) dx
    Vec_t phi(N_dofs);
    phi.setZero();
    PUM_ElemVec elvec_builder(fe_space, N_wave, k, f);
    lf::assemble::AssembleVectorLocally(0, dofh, elvec_builder, phi);

    const Eigen::SparseMatrix<Scalar> A_crs(A.makeSparse());

    // std::cout << A_crs << std::endl;
    // std::cout << phi << std::endl;

    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A_crs);
    Vec_t res;
    if(solver.info() == Eigen::Success) {
        res = solver.solve(phi);
    } else {
        LF_ASSERT_MSG(false, "Eigen Factorization failed")
    }
    return res;
}

double HE_PUM::L2_norm(size_type l, const Vec_t& mu) {
    auto mesh = mesh_hierarchy->getMesh(l);

    size_type N_wave(num_planwaves[l]);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), N_wave}});
    size_type N_dofs(dofh.NumDofs());

    lf::assemble::COOMatrix<Scalar> mass_matrix(N_dofs, N_dofs);
    
    // assemble for <grad(u), grad(v)>
    // (u, v) -> \int_K \alpha grad u grad v.conj + \gamma * u v.conj dx
    PUM_FEElementMatrix elmat_builder(N_wave, k, 0, 1.);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mass_matrix);
    
    const Eigen::SparseMatrix<Scalar> mass_mat = mass_matrix.makeSparse();
    double res = std::abs(mu.dot(mass_mat * mu.conjugate()));
    return std::sqrt(res);
}

double HE_PUM::H1_norm(size_type l, const Vec_t& mu) {
    auto mesh = mesh_hierarchy->getMesh(l);

    size_type N_wave(num_planwaves[l]);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), N_wave}});
    size_type N_dofs(dofh.NumDofs());

    lf::assemble::COOMatrix<Scalar> mass_matrix(N_dofs, N_dofs);
    
    // assemble for <grad(u), grad(v)>
    // (u, v) -> \int_K \alpha grad u grad v.conj + \gamma * u v.conj dx
    PUM_FEElementMatrix elmat_builder(N_wave, k, 1., 1.);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mass_matrix);
    
    const Eigen::SparseMatrix<Scalar> mass_mat = mass_matrix.makeSparse();
    double res = std::abs(mu.dot(mass_mat * mu.conjugate()));
    return std::sqrt(res);
}