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

std::pair<lf::assemble::COOMatrix<HE_PUM::Scalar>, HE_PUM::Vec_t> 
HE_PUM::build_equation(size_type level) {
    auto mesh = mesh_hierarchy->getMesh(level);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});
    
    // assemble for <grad(u), grad(v)> - k^2 uv
    PUM_FEElementMatrix elmat_builder(L, level, k);
    
    size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    // assemble boundary edge matrix, -i*k*u*v over \Gamma_R (outer boundary)
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

    PUM_EdgeMat edge_mat_builder(fe_space, outer_boundary, L, level, k);                                  
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector, \int_{\Gamma_R} gvdS
    Vec_t phi(N_dofs);
    phi.setZero();
    lf::mesh::utils::MeshFunctionGlobal mf_g{g};
    lf::mesh::utils::MeshFunctionGlobal mf_h{h};
    PUM_EdgeVec edgeVec_builder(fe_space, outer_boundary, L, level, k, g);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);

    return std::make_pair(A, phi);
}