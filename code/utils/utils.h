#pragma once

#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../Pum_WaveRay/HE_FEM.h"

using Scalar = std::complex<double>;
using size_type = unsigned int;
using coordinate_t = Eigen::Vector2d;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using triplet_t = Eigen::Triplet<Scalar>;
using SpMat_t = Eigen::SparseMatrix<Scalar>;
using FHandle_t = std::function<Scalar(const coordinate_t&)>;
using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

// Integer through entity e, \int_e fdx
Scalar LocalIntegral(const lf::mesh::Entity& e, int quad_degree, const FHandle_t& f);

// integrate over mesh
Scalar integrate(std::shared_ptr<lf::mesh::Mesh> mesh, const FHandle_t& f, int degree);

// L2 norm of the function, computed with the numerical quadrature in the given mesh
double L2_norm(std::shared_ptr<lf::mesh::Mesh> mesh, const FHandle_t& f, int degree);

// Test the manufacture solution, directly solve the equation in finest coarse.
void test_solve(HE_FEM& he_fem, const std::string& sol_name, 
    const std::string& output_folder, size_type L, const FHandle_t& u, 
    const FunGradient_t& grad_u);

void test_multigrid(HE_FEM& he_fem, int num_coarserlayer, const std::string& sol_name, 
    const std::string& output_folder, size_type L, const FHandle_t& u,
    const FunGradient_t& grad_u);

void print_save_error(std::vector<std::vector<double>>& data, 
    std::vector<std::string>& data_label, const std::string& sol_name, 
    const std::string& output_folder);

void tabular_output(std::vector<std::vector<double>>& data, 
    std::vector<std::string>& data_label, const std::string& sol_name, 
    const std::string& output_folder, bool save);

/*
 * Directional Gaussian Seidel relaxation.
 *   D.o.f. are first ordered according to the direction d they are associated with
 * 
 * Equation: Ax = \phi
 * u: initial guess of solution.
 * t: relaxation times
 * stride: number of plan waves
 * sol: true solution
 */
void Gaussian_Seidel(SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu);

void Gaussian_Seidel(SpMat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol, int stride);

// use the power iteration to compute the domainant eigenvalue of GS operator and 
// an associated eigenvector
std::pair<Vec_t, Scalar> power_GS(SpMat_t& A, int stride);

void Kaczmarz(SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu);

void v_cycle(Vec_t& u, Vec_t& f, std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, 
    std::vector<int>& stride, size_type mu1, size_type mu2);