#pragma once

#include <vector>
#include <functional>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

using namespace std::complex_literals;

/*
 * Helmholtz equation (HE):
 *  (\Laplace + k^2) u = 0 on \Omega
 *   \partial u / \partial n - iku = g in \Gamma_R
 *   u = h in \Gamma_D
 * 
 * This is the base class for solving HE by finite element method
 */
class HE_FEM {
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using coordinate_t = Eigen::Vector2d;
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

    HE_FEM(size_type levels, double wave_num, const std::string& mesh_path, FHandle_t g_, FHandle_t h_, bool hole):
        L(levels), k(wave_num), g(g_), h(h_), hole_exist(hole) {
        auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
        reader = std::make_shared<lf::io::GmshReader>(std::move(mesh_factory), mesh_path);
        mesh_hierarchy = lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(reader->mesh(), L);
    }

    // return mesh at l-th level
    std::shared_ptr<lf::mesh::Mesh> getmesh(size_type l) { return mesh_hierarchy->getMesh(l); }
    // equation: Ax=\phi, return (A, \phi)
    virtual std::pair<lf::assemble::COOMatrix<Scalar>, Vec_t> build_equation(size_type level) = 0; 

    // compute interesting error norms of ||uh - u||, u is the exact solution passed by function handler
    // and uh is a finite element solution and is represented by expansion coefficients.
    virtual double L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) = 0;
    virtual double H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) = 0;
    virtual double H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) = 0;

    // get the vector representation of function f
    virtual Vec_t fun_in_vec(size_type l, const FHandle_t& f) = 0;
    virtual size_type Dofs_perNode(size_type l) = 0;
    virtual lf::assemble::UniformFEDofHandler get_dofh(size_type l) = 0;

    virtual Mat_t prolongation(size_type l) = 0; // transfer operator: FE sapce l -> FE space {l+1}
    virtual Vec_t solve(size_type l) = 0;  // solve equaitons Ax=\phi on mesh l.
    // solve by multigrid method
    virtual void solve_multigrid(size_type start_layer, Vec_t& initial, int num_coarserlayer, 
        int mu1, int mu2) = 0; 
    virtual Vec_t power_multigird(size_type start_layer, int num_coarserlayer, int mu1, int mu2) = 0;

    // virtual std::pair<Vec_t, Scalar> power_multigrid(size_type l)

    virtual ~HE_FEM() = default;

protected:
    size_type L;  // number of refinement steps
    double k;  // wave number in the Helmholtz equation
    std::shared_ptr<lf::io::GmshReader> reader; // read the coarest mesh
    std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy;
    // mesh_hierarchy->getMesh(0) -- coarsest
    // mesh_hierarchy->getMesh(L) -- finest
    
    FHandle_t g;
    FHandle_t h;
    bool hole_exist; // whether the domain contains a hole inside
};