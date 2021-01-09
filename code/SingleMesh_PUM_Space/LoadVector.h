#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// linear form: l(v) = -\int_{\Gamma_R} gv dS(x)
class MassLoadVector{
public:
    using size_type = unsigned int;
    using vec_scalar = std::complex<double>;
    using elem_vec_t = Eigen::Matrix<mat_scalar, Eigen::Dynamic, 1>;
    using ElemMat = const elem_vec_t;
    
    using FHandle_t = std::function<std::complex<double>(const Eigen::Vector2d &)>;
    
    MassLoadVector(size_type L_, size_type l_, FHandle_t g_): L(L_), l(l_), g(g_){}
    
    virtual bool isActive(const lf::mesh::Entity & /*cell*/) { return true; }
    
    /*
     * @brief main routine for the computation of element matrices
     *
     * @param cell reference to the triangular cell for
     *        which the element matrix should be computed.
     * @return a square matrix with 3*2^(L-l+1) rows.
     */
    elem_mat_t Eval(const lf::mesh::Entity &edge);
private:
    size_type L;
    size_type l;
    FHandle_t g;
}

