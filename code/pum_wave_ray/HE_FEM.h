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
 * This is the basis class for solving HE by finite element methods
 */
class HE_FEM {
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;

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
    // compute the norm of a function represented by vector coefficient
    virtual double L2_norm(size_type l, const Vec_t& mu) = 0;
    virtual double H1_norm(size_type l, const Vec_t& mu) = 0;
    // get the vector representation of function f
    virtual Vec_t fun_in_vec(size_type l, const FHandle_t& f) = 0;
    virtual size_type Dofs_perNode(size_type l) = 0;
    virtual lf::assemble::UniformFEDofHandler get_dofh(size_type l) = 0;
    virtual ~HE_FEM() = default;

public:
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