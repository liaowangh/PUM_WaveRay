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

#include "../Pum/PUM_ElementMatrix.h"
#include "../ExtendPum/ExtendPUM_ElementMatrix.h"
#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../LagrangeO1/HE_LagrangeO1.h"
#include "../Pum_WaveRay/PUM_WaveRay.h"
#include "../ExtendPum_WaveRay/ExtendPUM_WaveRay.h"

/*
 * Function return the domain mass matrix \int_\Omega gamma * (u,v) dx
 * for LagrangeO1, Pum and ExtendPum
 */

SpMat_t O1_mass_mat(std::shared_ptr<lf::mesh::Mesh> mesh, Scalar gamma) {
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});

    lf::mesh::utils::MeshFunctionConstant<double> mf_alpha(0.0);
    lf::mesh::utils::MeshFunctionConstant<Scalar> mf_gamma(gamma);
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_alpha), decltype(mf_gamma)> 
    	elmat_builder(fe_space, mf_alpha, mf_gamma);
    
    int N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    return A.makeSparse();
}

SpMat_t PUM_mass_mat(std::shared_ptr<lf::mesh::Mesh> mesh, Scalar gamma, double k, int N_wave) {
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), N_wave}});

    PUM_ElementMatrix elmat_builder(N_wave, k, 0.0, gamma, 30);
    int N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    return A.makeSparse();
}

SpMat_t ePUM_mass_mat(std::shared_ptr<lf::mesh::Mesh> mesh, Scalar gamma, double k, int N_wave) {
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), N_wave + 1}});

    ExtendPUM_ElementMatrix elmat_builder(N_wave, k, 0.0, gamma, 30);
    int N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    return A.makeSparse();
}