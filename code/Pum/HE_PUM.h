#pragma once

#include <functional>
#include <vector>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../Pum_WaveRay/HE_FEM.h"

using namespace std::complex_literals;

/*
 * plan wave PUM spaces: {bi(x) * exp(ikdt x)}
 * W_l: {b_i^l(x) * e_t^l(x)}, N_l = 2^{L+1-l} 
 */
class HE_PUM: virtual public HE_FEM {
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SpMat_t = Eigen::SparseMatrix<Scalar>;
    using triplet_t = Eigen::Triplet<Scalar>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;

    /* Construcotr */
    HE_PUM(size_type levels, double wave_num, const std::string& mesh_path, 
        FHandle_t g, FHandle_t h, bool hole, std::vector<int> num_waves, int quad_degree=20): 
        HE_FEM(levels, wave_num, mesh_path, g, h, hole, num_waves, quad_degree){};

    /* functions inherit from base class */
    std::pair<lf::assemble::COOMatrix<Scalar>, Vec_t> build_equation(size_type level) override;
    double L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) override;
    double H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) override;
    double H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) override;
    Vec_t fun_in_vec(size_type l, const FHandle_t& f) override;

    size_type Dofs_perNode(size_type l) override { return num_planwaves[l]; }
    lf::assemble::UniformFEDofHandler get_dofh(size_type l) override {
        return lf::assemble::UniformFEDofHandler(getmesh(l), 
                {{lf::base::RefEl::kPoint(), Dofs_perNode(l)}});
    }

    SpMat_t prolongation(size_type l) override; // transfer operator: FE sapce l -> FE space {l+1}
    Vec_t solve(size_type l) override;  // solve equaitons Ax=\phi on mesh l.
    void solve_multigrid(Vec_t& initial, size_type start_layer, int num_coarserlayer, int nu1, int nu2, bool solve_coarest=false) override; 
    std::pair<Vec_t, Scalar> power_multigird(size_type start_layer, int num_coarserlayer, 
        int mu1, int mu2) override;

    /* functions specific to this class */
    double L2_BoundaryErr(size_type l, const Vec_t& mu, const FHandle_t& u,
        lf::mesh::utils::CodimMeshDataSet<bool> edge_selector);
    // Project the Dirichlet data on (h) into finite element space in Boundary
    Vec_t h_in_vec(size_type l, lf::mesh::utils::CodimMeshDataSet<bool> edge_selector,
        lf::mesh::utils::CodimMeshDataSet<bool> inner_point);
};