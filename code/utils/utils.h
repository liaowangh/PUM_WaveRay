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
#include "../SingleMesh_PUM_Space/HE_PUM.h"

using Scalar = std::complex<double>;
using size_type = unsigned int;
using coordinate_t = Eigen::Vector2d;
using vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using function_type = std::function<Scalar(const coordinate_t&)>;

// Integer through entity e, \int_e fdx
Scalar LocalIntegral(const lf::mesh::Entity& e, int quad_degree, const function_type& f);
// template <class MF>
// auto LocalIntegral(const lf::mesh::Entity &e, int quad_degree,
//                    const MF &mf) -> lf::mesh::utils::MeshFunctionReturnType<MF>;

// vector representation of function f
vec_t fun_in_vec(const lf::assemble::DofHandler& dofh, const function_type& f);

// compute the L2 norm giving the vector representation
double L2_norm(const lf::assemble::DofHandler&, const vec_t&);

double L2Err_norm(std::shared_ptr<lf::mesh::Mesh> mesh, const function_type& u, const vec_t& mu);

double H1_norm(const lf::assemble::DofHandler&, const vec_t&);

double H1_seminorm(const lf::assemble::DofHandler&, const vec_t&);

// Test the manufacture solution, directly solve the equation in finest coarse.
void solve_directly(HE_FEM& he_fem, const std::string& sol_name, size_type L, const function_type& u);