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
#include "../planwave_pum/HE_PUM.h"

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
class PUM_WaveRay{
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using coordinate_t = Eigen::Vector2d;
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

    PUM_WaveRay(size_type levels, double wave_number, const std::string& mesh_path, 
        FHandle_t g, FHandle_t h, bool hole): L(levels), k(wave_number) {
        num_planwaves = std::vector<int>(levels+1);
        num_planwaves[levels] = 2;
        for(int i = levels; i > 0; --i) {
            num_planwaves[i-1] = 2 * num_planwaves[i];
        }

        Lagrange_fem = std::make_shared<HE_LagrangeO1>(levels, wave_number, mesh_path, g, h, hole);
        planwavePUM_fem = std::make_shared<HE_PUM>(levels, wave_number, mesh_path, g, h, num_planwaves, hole);
    }

    // std::pair<lf::assemble::COOMatrix<Scalar>, Vec_t> build_equation(size_type level) override; 

    // compute interesting error norms of ||uh - u||, u is the exact solution passed by function handler
    // and uh is a finite element solution and is represented by expansion coefficients.
    // double L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) override;
    // double H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) override;
    // double H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) override;

    // // get the vector representation of function f
    // Vec_t fun_in_vec(size_type l, const FHandle_t& f) override;
    // size_type Dofs_perNode(size_type l) override { return l == L ? 1 : num_planwaves[l]; };
    // lf::assemble::UniformFEDofHandler get_dofh(size_type l) override {
    //     return lf::assemble::UniformFEDofHandler(getmesh(l), 
    //             {{lf::base::RefEl::kPoint(), Dofs_perNode(l)}});
    // }

    // prolongation and restriction operator for S_l and E_l, actually for latter, 
    // the prolongation and restriction should be swaped, use the current name just to 
    // be consistent with S_l.
    void Prolongation_Lagrange();
    void Prolongation_planwave();
    // void Restriction_planwave();
    void Prolongation_SE();
    void Prolongation_SE_S();

    void v_cycle(Vec_t& initial, size_type start_layer, size_type num_wavelayer,
        size_type mu1, size_type mu2);

public:  
    size_type L;
    double k;
    std::vector<int> num_planwaves; // number of plan waves in coarser mesh
    std::shared_ptr<HE_LagrangeO1> Lagrange_fem;
    std::shared_ptr<HE_PUM> planwavePUM_fem;

    std::vector<Mat_t> P_Lagrange; // prolongation operator between Lagrange FE spaces, S_l -> S_{l+1}
    std::vector<Mat_t> P_planwave; // prolongation operator between plan wave spaces, E_l->E_{l+1}   

    // std::vector<Mat_t> R_Lagrange; // S_{l+1} -> S_l
    // std::vector<Mat_t> R_planwave; // E_{l+1} -> E_l, note that E_{l+1} is a subset of E_l

    std::vector<Mat_t> P_SE;       // SxE_l -> SxE_{l+1}
    std::vector<Mat_t> P_SE_S;     // SxE_l -> S_{l+1}
};

/*
 * Directional Gaussian Seidel relaxation.
 *   D.o.f. are first ordered according to the direction d they are associated with
 * 
 * Equation: Ax = \phi
 * u: initial guess of solution.
 * t: relaxation times
 * stride: number of plan waves
 */
template <typename mat_type>
void Gaussian_Seidel(mat_type& A, PUM_WaveRay::Vec_t& phi, PUM_WaveRay::Vec_t& u, int stride, int t){
    // u: initial value; t: number of iterations
    int N = A.rows();
    for(int i = 0; i < t; ++i){
        for(t = 0; t < stride; ++t) {
            // direction
            for(int k = 0; k < N / stride; ++k) {
                int j = k * stride + t;
                std::complex<double> tmp = A.row(j) * u;

                // if(i == 0){
                //     std::cout << i << " " << j << std::endl;
                //     std::cout << phi(j) - tmp << std::endl;
                // }
                
                u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
            }
        }
    }
}

