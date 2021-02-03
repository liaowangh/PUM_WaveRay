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

#include "../pum_wave_ray/HE_FEM.h"
#include "../planwave_pum/HE_PUM.h"

using Scalar = std::complex<double>;
using size_type = unsigned int;
using coordinate_t = Eigen::Vector2d;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using FHandle_t = std::function<Scalar(const coordinate_t&)>;
using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

// Integer through entity e, \int_e fdx
Scalar LocalIntegral(const lf::mesh::Entity& e, int quad_degree, const FHandle_t& f);

// Test the manufacture solution, directly solve the equation in finest coarse.
void test_solve(HE_FEM& he_fem, const std::string& sol_name, 
    const std::string& output_folder, size_type L, const FHandle_t& u, 
    const FunGradient_t& grad_u);

void print_save_error(std::vector<int>& N, std::vector<std::vector<double>>& data, 
    std::vector<std::string>& err_str, const std::string& sol_name, 
    const std::string& output_folder);

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
void Gaussian_Seidel(Mat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu);

void Gaussian_Seidel(Mat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol, int stride);

// use the power iteration to compute the domainant eigenvalue of GS operator and 
// an associated eigenvector
std::pair<Vec_t, Scalar> power_GS(Mat_t& A, int stride);

void v_cycle(Vec_t& u, Vec_t& f, std::vector<Mat_t>& Op, std::vector<Mat_t>& I, 
    std::vector<int>& stride, size_type mu1, size_type mu2);