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

#include "HE_FEM.h"

using namespace std::complex_literals;

/*
 * When using the PUM wary ray method to solve the Helmholtz equation,
 * The finite element space associated with the finest mesh is S1
 * The FE spaces in coarser mesh are PUM spaces {bi(x) * exp(ikdt x}
 */
class PUM_WaveRay: public HE_FEM {
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d&)>;

    PUM_WaveRay(size_type L_, double k_, std::string mesh_path, FHandle_t g_, FHandle_t h_, bool hole):
        HE_FEM(L_, k_, mesh_path, g_, h_, hole){}; // constructor 

    lf::assemble::UniformFEDofHandler get_dofh(size_type);
    
    std::pair<lf::assemble::COOMatrix<Scalar>, Vec_t> build_equation(size_type level) override; 
    double L2_norm(size_type l, const Vec_t& mu) override;
    double H1_norm(size_type l, const Vec_t& mu) override;
    Vec_t fun_in_vec(size_type l, const FHandle_t& f) override;

    size_type Dofs_perNode(size_type l) { return 1 ;};

    void Prolongation_LF(); // generate P_LF
    void Prolongation_PW(); // generate P_PW
    void Restriction_PW();  // generate R_PW

    Mat_t Prolongation_PUM(int l); // level l -> level l+1 in PUM spaces
    Mat_t Restriction_PUM(int l);  // level l+1 -> level l in PUM spaces
    
    Scalar integration_mesh(int level, PUM_WaveRay::FHandle_t f);

    void v_cycle(Vec_t& u, size_type mu1, size_type mu2); // initial: u, relaxation times: mu1 and mu2

public:
    std::vector<Mat_t> P_LF; // P_LF[i]: level i -> level i+1, prolongation of Lagrangian FE spaces
    std::vector<Mat_t> P_PW; // P_PW[i]: level i -> level i+1, planar wave spaces
    
    std::vector<Mat_t> R_PW; // R_PW[i]: level i+1 -> level i, planar wave spaces
};

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

/*
 * Relaxation using Gaussian Seidel iteration
 * Equation: Ax = \phi
 * u: initial guess of solution.
 * t: relaxation times
 */
template <typename mat_type>
void Gaussian_Seidel(mat_type& A, PUM_WaveRay::Vec_t& phi, PUM_WaveRay::Vec_t& u, int t);

