#pragma once

#include <functional>
#include <vector>
#include <memory>

#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "HE_FEM.h"
#include "../LagrangeO1/HE_LagrangeO1.h"
#include "../Pum/HE_PUM.h"

using namespace std::complex_literals;

/*
 * When using the PUM wary ray method to solve the Helmholtz equation,
 * The finite element space associated with the finest mesh is LagrangeO1
 * The FE spaces in coarser mesh are plan wave PUM spaces {bi(x) * exp(ikdt x}
 * 
 * FE spaces:
 *  S_i : Lagrange FE in mesh i
 *  E_i : plan wave spaces {exp(ikd x)}
 * 
 * Some notation:
 *  Nl  = 2^{N+1-l} = num_planwaves[l]
 *  dtl = [cos(2pi * t/Nl), sin(2pi * t/Nl)] 
 *  etl = exp(ik* dtl x)
 */
class PUM_WaveRay: public HE_LagrangeO1, public HE_PUM {
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using coordinate_t = Eigen::Vector2d;
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using triplet_t = Eigen::Triplet<Scalar>;
    using SpMat_t = Eigen::SparseMatrix<Scalar>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

    PUM_WaveRay(size_type levels, double wave_number, const std::string& mesh_path, 
        FHandle_t g, FHandle_t h, bool hole, std::vector<int> num_waves, int quad_degree=20): 
            HE_FEM(levels, wave_number, mesh_path, g, h, hole, num_waves, quad_degree),
            HE_LagrangeO1(levels, wave_number, mesh_path, g, h, hole),
            HE_PUM(levels, wave_number, mesh_path, g, h, hole, num_waves) {}

    std::pair<lf::assemble::COOMatrix<Scalar>, Vec_t> build_equation(size_type level) override; 

    // compute interesting error norms of ||uh - u||, u is the exact solution passed by function handler
    // and uh is a finite element solution and is represented by expansion coefficients.
    double L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) override;
    double H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) override;
    double H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) override;

    // get the vector representation of function f
    Vec_t fun_in_vec(size_type l, const FHandle_t& f) override;
    size_type Dofs_perNode(size_type l) override { return l == L ? 1 : num_planwaves[l]; };
    lf::assemble::UniformFEDofHandler get_dofh(size_type l) override {
        return lf::assemble::UniformFEDofHandler(getmesh(l), 
                {{lf::base::RefEl::kPoint(), Dofs_perNode(l)}});
    }

    SpMat_t prolongation(size_type l) override;
    Vec_t solve(size_type l) override;
    void solve_multigrid(Vec_t& initial, size_type start_layer, int num_wavelayer, int nu1, int nu2, bool solve_coarest = false) override;
    std::pair<Vec_t, Scalar> power_multigird(size_type start_layer, int num_coarserlayer, 
        int mu1, int mu2) override;
};



