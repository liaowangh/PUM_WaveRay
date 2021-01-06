#include "utils.h"
#include "../pum_wave_ray/pum_fem.h"

vec_t fun_in_vec(const lf::assemble::DofHandler& dofh, const function_type& f) {
    size_type N_dofs(dofh.NumDofs());
    vec_t res(N_dofs);
    for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
        const lf::mesh::Entity& dof_node{dofh.Entity(dofnum)};
        const coordinate_t node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
        res(dofnum) = f(node_pos);
    }
    return res;
}

double L2_norm(const lf::assemble::DofHandler& dofh, const vec_t& u) {
    double res = 0.0;
    int N_dofs = dofh.NumDofs();
    lf::assemble::COOMatrix<double> mass_matrix(N_dofs, N_dofs);
    
    lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
    lf::mesh::utils::MeshFunctionConstant<double> mf_zero(0);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(dofh.Mesh());
    
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_zero), decltype(mf_identity)>
        elmat_builder(fe_space, mf_zero, mf_identity);
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mass_matrix);
    
    const Eigen::SparseMatrix<double> mass_mat = mass_matrix.makeSparse();
    res = std::sqrt(std::abs(u.dot(mass_mat * u.conjugate())));
    return res;
}

double H1_norm(const lf::assemble::DofHandler& dofh, const vec_t& u) {
    double res = 0.0;
    int N_dofs = dofh.NumDofs();
    lf::assemble::COOMatrix<double> mass_matrix(N_dofs, N_dofs);
    
    lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(dofh.Mesh());
    
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_identity)>
        elmat_builder(fe_space, mf_identity, mf_identity);
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mass_matrix);
    
    const Eigen::SparseMatrix<double> mass_mat = mass_matrix.makeSparse();
    res = std::sqrt(std::abs(u.dot(mass_mat * u.conjugate())));
    return res;
}

void solve_directly(const std::string& mesh_path, size_type L, double wave_num, const function_type& u, const function_type& g, const function_type& h) {
    PUM_FEM pum_fem(L, wave_num, mesh_path, g, h);
    
    std::vector<int> ndofs;
    std::vector<double> L2err, H1err;
    
    for(size_type level = 0; level <= L; ++level) {
        auto eq_pair = pum_fem.build_equation(level);

        // solve linear system of equationx A*x = \phi
        const Eigen::SparseMatrix<Scalar> A_crs(eq_pair.first.makeSparse());
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(A_crs);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> appro_vec;
        if(solver.info() == Eigen::Success) {
            appro_vec = solver.solve(eq_pair.second);
        } else {
            LF_ASSERT_MSG(false, "Eigen Factorization failed")
        }

        // find the true vector representation
        auto dofh = lf::assemble::UniformFEDofHandler(pum_fem.getmesh(level), {{lf::base::RefEl::kPoint(), 1}});
        vec_t true_vec = fun_in_vec(dofh, u);

        double l2_err = L2_norm(dofh, appro_vec - true_vec);
        double h1_err = H1_norm(dofh, appro_vec - true_vec);
        
        ndofs.push_back(dofh.NumDofs());
        L2err.push_back(l2_err);
        H1err.push_back(h1_err);
    }
    
    // Tabular output of the results
    
    std::cout << std::left << std::setw(10) << "N" << std::setw(20)
              << "L2 err" << std::setw(20) << "H1 err" << std::endl;
    for (int l = 0; l <= L; ++l) {
      std::cout << std::left << std::setw(10) << ndofs[l] << std::setw(20)
                << L2err[l] << std::setw(20) << H1err[l] << std::endl;
    }
}
