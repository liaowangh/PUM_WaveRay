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
    
    void Prolongation_P(); // generate P
    void Prolongation_Q(); // generate Q
    
    std::pair<lf::assemble::COOMatrix<mat_scalar>, rhs_vec_t> build_equation(size_type level); // equation: Ax=\phi, return (A, \phi)
    mat_scalar integration_mesh(int level, function_type f);
    // use three point quadtature rule to approximate \int_fdx based on mesh_hierarchy->getMesh(level)
    
// private:
    size_type L_;  // number of refinement steps
    double k_;  // wave number in the Helmholtz equation
    std::shared_ptr<lf::io::GmshReader> reader; // read the coarest mesh
    std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy;
    // mesh_hierarchy->getMesh(0) -- coarsest
    // mesh_hierarchy->getMesh(L) -- finest
    
    std::vector<elem_mat_t> P; // P[i]: level i -> level i+1, standard space
    std::vector<elem_mat_t> Q; // Q[i]: level i -> level i+1, plane wave
    
    function_type g_;
    function_type h_;
};


/*
void Gaussian_Seidel(PUM_FEM::elem_mat_t& A, PUM_FEM::rhs_vec_t& phi, PUM_FEM::rhs_vec_t& u, int t) {
    // u: initial value; t: number of iterations
    int N = A.rows();
    for(int i = 0; i < t; ++i){
        for(int j = 0; j < N; ++j) {
            auto tmp = A.row(j).dot(u);
            u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
        }
    }
}
*/