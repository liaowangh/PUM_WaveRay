#include <fstream>
#include <string>

#include "utils.h"
#include "../pum_wave_ray/HE_FEM.h"
#include "../pum_wave_ray/PUM_WaveRay.h"

Scalar LocalIntegral(const lf::mesh::Entity& e, int quad_degree, const FHandle_t& f) {
    auto qr = lf::quad::make_QuadRule(e.RefEl(), quad_degree);
    auto global_points = e.Geometry()->Global(qr.Points());
    auto weights_ie = (qr.Weights().cwiseProduct(e.Geometry()->IntegrationElement(qr.Points()))).eval();

    Scalar temp = 0.0;
    for (Eigen::Index i = 0; i < qr.NumPoints(); ++i) {
        temp += weights_ie(i) * f(global_points.col(i));
    }
    return temp;

}

/*
 template <class MF>
 auto LocalIntegral(const lf::mesh::Entity &e, int quad_degree,
                    const MF &mf) -> lf::mesh::utils::MeshFunctionReturnType<MF> {
    using MfType = lf::mesh::utils::MeshFunctionReturnType<MF>;

    auto qr = lf::quad::make_QuadRule(e.RefEl(), quad_degree);

    auto values = mf(e, qr.Points());
    auto weights_ie =
        (qr.Weights().cwiseProduct(e.Geometry()->IntegrationElement(qr.Points())))
            .eval();
    LF_ASSERT_MSG(values.size() == qr.NumPoints(),
                    "mf returns vector with wrong size.");
    if constexpr (std::is_arithmetic_v<MfType>) {  // NOLINT
        auto value_m = Eigen::Map<Eigen::Matrix<MfType, 1, Eigen::Dynamic>>(
            &values[0], 1, values.size());
        return (value_m * weights_ie)(0);
    }
    
    if constexpr (lf::base::is_eigen_matrix<MfType>) {  // NOLINT
        constexpr int size = MfType::SizeAtCompileTime;
        if constexpr (size != Eigen::Dynamic) {
        auto value_m = Eigen::Map<
            Eigen::Matrix<typename MfType::Scalar, size, Eigen::Dynamic>>(
            &values[0](0, 0), size, values.size());
        MfType result;
        auto result_m =
            Eigen::Map<Eigen::Matrix<typename MfType::Scalar, size, 1>>(
                &result(0, 0));
        result_m = value_m * weights_ie;
        return result;
        }
    }
    // fallback: we cannot make any optimizations:
    MfType temp = weights_ie(0) * values[0];
    for (Eigen::Index i = 1; i < qr.NumPoints(); ++i) {
        temp = temp + weights_ie(i) * values[i];
    }
    return temp;
}
*/

Vec_t fun_in_vec(const lf::assemble::DofHandler& dofh, const FHandle_t& f) {
    size_type N_dofs(dofh.NumDofs());
    Vec_t res(N_dofs);
    for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
        const lf::mesh::Entity& dof_node{dofh.Entity(dofnum)};
        const coordinate_t node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
        res(dofnum) = f(node_pos);
    }
    return res;
}

/*
 * compute norm rely on the mass matrix which is the finite element Galerkin matrix 
 * for the L2 inner product (u,v)->\int_\Omega uvdx 
 */
double L2_norm(const lf::assemble::DofHandler& dofh, const Vec_t& u) {
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

/*
 * compute ||u-mu||_2, 
 * where u (manufacture solution) is of FHandle_t,
 * solution is giving by vector representation mu
 */ 
double L2Err_norm(std::shared_ptr<lf::mesh::Mesh> mesh, const FHandle_t& u, const Vec_t& mu) {
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    // u has to be wrapped into a mesh function for error computation
    lf::mesh::utils::MeshFunctionGlobal mf_u{u};
    // create mesh function representing solution 
    auto mf_mu = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu);

    auto u_conj = [&u](const coordinate_t& x) -> Scalar {
        return std::conj(u(x));};
    lf::mesh::utils::MeshFunctionGlobal mf_u_conj{u_conj};
    auto mf_mu_conj = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu.conjugate());
    
    auto mf_square = (mf_u - mf_mu) * (mf_u_conj - mf_mu_conj);
    double L2err = std::abs(lf::uscalfe::IntegrateMeshFunction(*mesh, mf_square, 5));
    return std::sqrt(L2err);
}

double H1_norm(const lf::assemble::DofHandler& dofh, const Vec_t& u) {
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

double H1_seminorm(const lf::assemble::DofHandler& dofh, const Vec_t& u) {
    double res = 0.0;
    int N_dofs = dofh.NumDofs();
    lf::assemble::COOMatrix<double> mass_matrix(N_dofs, N_dofs);
    
    lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
    lf::mesh::utils::MeshFunctionConstant<double> mf_zero(0);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(dofh.Mesh());
    
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_zero)>
        elmat_builder(fe_space, mf_identity, mf_zero);
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mass_matrix);
    
    const Eigen::SparseMatrix<double> mass_mat = mass_matrix.makeSparse();
    res = std::sqrt(std::abs(u.dot(mass_mat * u.conjugate())));
    return res;
}

void solve_directly(HE_FEM& he_fem, const std::string& sol_name, size_type L, 
                    const FHandle_t& u, const FunGradient_t& grad_u) {
    std::vector<int> ndofs;
    std::vector<double> L2err, H1serr, H1err;
    
    for(size_type level = 0; level <= L; ++level) {
        auto eq_pair = he_fem.build_equation(level);

        // solve linear system of equations A*x = \phi
        const Eigen::SparseMatrix<Scalar> A_crs(eq_pair.first.makeSparse());
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(A_crs);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> fe_sol;
        if(solver.info() == Eigen::Success) {
            fe_sol = solver.solve(eq_pair.second);
        } else {
            LF_ASSERT_MSG(false, "Eigen Factorization failed")
        }
        
        double l2_err = he_fem.L2_Err(level, fe_sol, u);
        double h1_serr = he_fem.H1_semiErr(level, fe_sol, grad_u);
        double h1_err = std::sqrt(l2_err*l2_err + h1_serr*h1_serr);
        
        ndofs.push_back(eq_pair.second.size());
        L2err.push_back(l2_err);
        H1serr.push_back(h1_serr);
        H1err.push_back(h1_err);
    }
    
    // Tabular output of the results
    std::cout << sol_name << std::endl;
    std::cout << std::left << std::setw(10) << "N" << std::setw(20)
              << "L2_err" << std::setw(20) << "H1_serr" << std::setw(20) 
              << "H1_err" << std::endl;
    for (int l = 0; l <= L; ++l) {
      std::cout << std::left << std::setw(10) << ndofs[l] << std::setw(20)
                << L2err[l] << std::setw(20) << H1serr[l] << std::setw(20)
                << H1err[l] << std::endl;
    }
    
    // write the result to the file
    std::string output_file = "../plot_err/" + sol_name + ".txt";
    std::ofstream out(output_file);
    out << "N " << "L2_err " << "H1_err" << std::endl;
    for(int l = 0; l <= L; ++l) {
        out << ndofs[l] << " " << L2err[l] << " " << H1err[l] << std::endl;
    } 
}
