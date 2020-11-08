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

class PUM_FEElementMatrix{
public:
    using size_type = unsigned int;
    using mat_scalar = std::complex<double>;
    using elem_mat_t = Eigen::Matrix<mat_scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ElemMat = const elem_mat_t;
    
    PUM_FEElementMatrix(size_type L_, size_type l_, double k_): L(L_), l(l_), k(k_){}
    
    virtual bool isActive(const lf::mesh::Entity & /*cell*/) { return true; }
    
    /*
     * @brief main routine for the computation of element matrices
     *
     * @param cell reference to the triangular cell for
     *        which the element matrix should be computed.
     * @return a square matrix with 3*2^(L-l+1) rows.
     */
    elem_mat_t Eval(const lf::mesh::Entity &cell);
private:
    size_type L;
    size_type l;
    double k;
}

