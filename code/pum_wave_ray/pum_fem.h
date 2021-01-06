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

class exp_wave{
// f(x,y) = exp(i(d1x+d2y))
public:
    exp_wave(double d1, double d2): d1_(d1), d2_(d2) {}
    std::complex<double> operator()(Eigen::Vector2d x){
        return std::exp(1i * (d1_ * x(0) + d2_ * x(1)));
    }
private:
    double d1_, d2_; // frequency
};

class PUM_FEM {
public:
    using size_type = unsigned int;
    using mat_scalar = std::complex<double>;
    using elem_mat_t = Eigen::Matrix<mat_scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using rhs_vec_t = Eigen::Matrix<mat_scalar, Eigen::Dynamic, 1>;
    using function_type = std::function<mat_scalar(Eigen::Vector2d)>;
    
    PUM_FEM(size_type L, double k, std::string mesh_path, function_type g, function_type h); // constructor 
    
    lf::assemble::UniformFEDofHandler generate_dof(size_type);
    
    // return mesh at l-th level
    std::shared_ptr<lf::mesh::Mesh> getmesh(size_type l) { return mesh_hierarchy->getMesh(l); }
    
    void Prolongation_LF(); // generate P_LF
    void Prolongation_PW(); // generate P_PW
    void Restriction_PW();  // generate R_PW

    elem_mat_t Prolongation_PUM(int l); // level l -> level l+1 in PUM spaces
    elem_mat_t Restriction_PUM(int l);  // level l+1 -> level l in PUM spaces
    
    std::pair<lf::assemble::COOMatrix<mat_scalar>, rhs_vec_t> build_equation(size_type level); // equation: Ax=\phi, return (A, \phi)
    mat_scalar integration_mesh(int level, function_type f);
    
    void v_cycle(rhs_vec_t& u, size_type mu1, size_type mu2); // initial: u, relaxation times: mu1 and mu2

// private:
    size_type L_;  // number of refinement steps
    double k_;  // wave number in the Helmholtz equation
    std::shared_ptr<lf::io::GmshReader> reader; // read the coarest mesh
    std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy;
    // mesh_hierarchy->getMesh(0) -- coarsest
    // mesh_hierarchy->getMesh(L) -- finest
    
    std::vector<elem_mat_t> P_LF; // P_LF[i]: level i -> level i+1, prolongation of Lagrangian FE spaces
    std::vector<elem_mat_t> P_PW; // P_PW[i]: level i -> level i+1, planar wave spaces
    
    std::vector<elem_mat_t> R_PW; // R_PW[i]: level i+1 -> level i, planar wave spaces

    function_type g_;
    function_type h_;
};


/*
 * Relaxation using Gaussian Seidel iteration
 * Equation: Ax = \phi
 * u: initial guess of solution.
 * t: relaxation times
 */
template <typename mat_type>
void Gaussian_Seidel(mat_type& A, PUM_FEM::rhs_vec_t& phi, PUM_FEM::rhs_vec_t& u, int t);

