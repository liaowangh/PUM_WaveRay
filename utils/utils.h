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

// return {k, b}, such that y = kx + b
std::vector<double> linearFit(const std::vector<double> x, const std::vector<double> y);

// L2 norm of the function, computed with the numerical quadrature in the given mesh
double L2_norm(std::shared_ptr<lf::mesh::Mesh> mesh, const FHandle_t& f, int degree);

// Test the manufacture solution, directly solve the equation in finest coarse.
void test_solve(HE_FEM& he_fem, const std::string& sol_name, 
    const std::string& output_folder, int L, const FHandle_t& u, 
    const FunGradient_t& grad_u);

template <typename data_type>
void print_save_error(std::vector<std::vector<data_type>>& data, 
    std::vector<std::string>& data_label, const std::string& sol_name, 
    const std::string& output_folder) {
    
    std::cout << sol_name << std::endl;
    std::cout << std::left;
    for(int i = 0; i < data_label.size(); ++i){
        std::cout << std::setw(15) << data_label[i];
    }
    std::cout << std::endl;
    std::cout << std::left << std::scientific << std::setprecision(1);
    for(int l = 0; l < data[0].size(); ++l) {
        for(int i = 0; i < data.size(); ++i) {
            std::cout << std::setw(15) << data[i][l];
        }
        std::cout << std::endl;
    }

    // write the result to the file
    std::string output_file = output_folder + sol_name + ".txt";
    std::ofstream out(output_file);

    out << data_label[0];
    for(int i = 1; i < data_label.size(); ++i) {
        out << " " << data_label[i];
    }
    out << std::endl;
    for(int l = 0; l < data[0].size(); ++l) {
        out << data[0][l];
        for(int i = 1; i < data.size(); ++i) {
            out << " " << data[i][l];
        }
        out << std::endl;
    } 
}

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
// void Gaussian_Seidel(const SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu);
void Gaussian_Seidel(SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu);

void Gaussian_Seidel(SpMat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol, int stride);

// void block_GS(const SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu);
void block_GS(SpMat_t& A, Vec_t& phi, Vec_t& u, int stride, int mu);

void block_GS(SpMat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol, int stride);

void Kaczmarz(SpMat_t& A, Vec_t& phi, Vec_t& u, int mu);

void Kaczmarz(SpMat_t& A, Vec_t& phi, Vec_t& u, Vec_t& sol);

// use the power iteration to compute the domainant eigenvalue of GS operator and 
// an associated eigenvector
std::pair<Vec_t, Scalar> power_GS(SpMat_t& A, int stride);
std::pair<Vec_t, Scalar> power_GS(Mat_t& A, int stride);
std::pair<Vec_t, Scalar> power_block_GS(SpMat_t& A, int stride);
std::pair<Vec_t, Scalar> power_kaczmarz(SpMat_t& A);

void v_cycle(Vec_t& u, Vec_t& f, std::vector<SpMat_t>& Op, std::vector<SpMat_t>& I, 
    std::vector<int>& stride, int mu1, int mu2, bool solve_on_coarest=true);

void mg_factor(HE_FEM& he_fem, int L, int nr_coarsemesh, double k, 
    std::vector<int>& stride, FHandle_t u, bool solve_coarest);

std::pair<Vec_t, Scalar> power_multigird(HE_FEM& he_fem, int start_layer, 
    int num_coarserlayer, std::vector<int>& stride, int nu1, int nu2, bool verbose);