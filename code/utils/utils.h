#pragma once

#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../pum_wave_ray/HE_FEM.h"
#include "../planwave_pum/HE_PUM.h"

using Scalar = std::complex<double>;
using size_type = unsigned int;
using coordinate_t = Eigen::Vector2d;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using FHandle_t = std::function<Scalar(const coordinate_t&)>;
using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

// Integer through entity e, \int_e fdx
Scalar LocalIntegral(const lf::mesh::Entity& e, int quad_degree, const FHandle_t& f);
// template <class MF>
// auto LocalIntegral(const lf::mesh::Entity &e, int quad_degree,
//                    const MF &mf) -> lf::mesh::utils::MeshFunctionReturnType<MF>;

// vector representation of function f
Vec_t fun_in_vec(const lf::assemble::DofHandler& dofh, const FHandle_t& f);

// compute the L2 norm giving the vector representation
double L2_norm(const lf::assemble::DofHandler&, const Vec_t&);

double L2Err_norm(std::shared_ptr<lf::mesh::Mesh> mesh, const FHandle_t& u, const Vec_t& mu);

double H1_norm(const lf::assemble::DofHandler&, const Vec_t&);

double H1_seminorm(const lf::assemble::DofHandler&, const Vec_t&);

// Test the manufacture solution, directly solve the equation in finest coarse.
void solve_directly(HE_FEM& he_fem, const std::string& sol_name, size_type L, 
                    const FHandle_t& u, const FunGradient_t& grad_u);