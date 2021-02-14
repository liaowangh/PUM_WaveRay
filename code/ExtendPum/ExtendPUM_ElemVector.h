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

/*
 * (local) Linear form: l(v) = \int_K f v.conjdS(x),
 * v = bi * exp(ikdt x), v.conj = bi * exp(-ikdt x)
 */
class ExtendPUM_ElemVec{
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    
    ExtendPUM_ElemVec(std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
        size_type N, double k, FHandle_t f, int degree=20): 
        fe_space_(fe_space), N_(N), k_(k), f_(f), degree_(degree){}
    
    bool isActive(const lf::mesh::Entity& cell) { return true; }
    
    Vec_t Eval(const lf::mesh::Entity &cell);
private:
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_;
    size_type N_; // number of planner waves
    double k_;
    FHandle_t f_;
    int degree_;
};

