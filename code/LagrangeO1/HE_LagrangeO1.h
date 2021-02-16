#pragma once

#include <functional>
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

#include "../Pum_WaveRay/HE_FEM.h"

using namespace std::complex_literals;

/*
 * Solve the Helmholtz equation in Lagrange Finite element space (O1)
 * 
 * FE spaces:
 *  S_i: Lagrange FE in mesh i
 */
class HE_LagrangeO1: virtual public HE_FEM {
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SpMat_t = Eigen::SparseMatrix<Scalar>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

    HE_LagrangeO1(size_type levels, double wave_num, const std::string& mesh_path, 
        FHandle_t g, FHandle_t h, bool hole, int quad_degree=20): 
        HE_FEM(levels, wave_num, mesh_path, g, h, hole,
            std::vector<int>(levels+1, 1), quad_degree){};

    std::pair<lf::assemble::COOMatrix<Scalar>, Vec_t> build_equation(size_type l) override;
    double L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) override;
    double H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) override;
    double H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) override;
    Vec_t fun_in_vec(size_type l, const FHandle_t& f) override;

    lf::assemble::UniformFEDofHandler get_dofh(size_type l) override {
        return lf::assemble::UniformFEDofHandler(getmesh(l), 
                {{lf::base::RefEl::kPoint(), Dofs_perNode(l)}});
    }
    size_type Dofs_perNode(size_type l) override { return 1; }

    SpMat_t prolongation(size_type l) override;
    Vec_t solve(size_type l) override;
    Vec_t solve_multigrid(size_type start_layer, int num_coarserlayer, int mu1, int mu2) override;
    std::pair<Vec_t, Scalar> power_multigird(size_type start_layer, int num_coarserlayer, 
        int mu1, int mu2) override;
};