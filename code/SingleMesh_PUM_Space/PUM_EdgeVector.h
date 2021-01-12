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

// linear form: l(v) = \int_{\Gamma_R} gv dS(x)
class PUM_EdgeVec{
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using EdgeVec_t = Eigen::Matrix<mat_scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    
    PUM_EdgeVec(std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_,
        lf::mesh::utils::CodimMeshDataSet<bool> &edge_selector_, size_type L_, size_type l_, FHandle_t g_): 
        fe_space(fe_space_), edge_selector(edge_selector_), L(L_), l(l_), g(g_){}
    
    bool isActive(const lf::mesh::Entity& edge) { return edge_selector(ege); }
    
    EdgeVec_t Eval(const lf::mesh::Entity &edge);
private:
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space;
    lf::mesh::utils::CodimMeshDataSet<bool> &edge_selector;
    size_type L;
    size_type l;
    FHandle_t g;
}

