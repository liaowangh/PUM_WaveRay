#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// linear form: l(v) = \int_K fv dS(x)
class PUM_ElemVec{
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using ElemVec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    
    PUM_ElemVec(std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_,
        size_type L_, size_type l_, double k_, FHandle_t f_): fe_space(fe_space_), L(L_), l(l_), k(k_), f(f_){}
    
    bool isActive(const lf::mesh::Entity& cell) { return true; }
    
    ElemVec_t Eval(const lf::mesh::Entity &cell);
private:
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space;
    size_type L;
    size_type l;
    double k;
    FHandle_t f;
};

