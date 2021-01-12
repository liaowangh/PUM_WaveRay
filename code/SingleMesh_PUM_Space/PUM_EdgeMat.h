#pragma once

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

class PUM_EdgeMat {
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using ElemMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    
    PUM_EdgeMat(std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_, 
        lf::mesh::utils::CodimMeshDataSet<bool> &edge_selector_, size_type L_, size_type l_, double k_): 
        fe_sapce(fe_space_), edge_selector(edge_selector_), L(L_), l(l_), k(k_) {}
    virtual bool isActive(const lf::mesh::Entity &edge) {
        return edge_selector(edge);
    }
    
    ElemMat Eval(const lf::mesh::Entity &edge);
private:
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space;
    lf::mesh::utils::CodimMeshDataSet<bool> &edge_selector;
    size_type L;
    size_type l;
    double k;
};
