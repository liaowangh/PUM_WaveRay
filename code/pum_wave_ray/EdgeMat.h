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

class EdgeMatProvider {
public:
    using size_type = unsigned int;
    using mat_scalar = std::complex<double>;
    using elem_mat_t = Eigen::Matrix<mat_scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ElemMat = const elem_mat_t;
    
    EdgeMatProvider(lf::mesh::utils::CodimMeshDataSet<bool> &bd_flags_, size_type L_, size_type l_, double k_): bd_flags(bd_flags_), L(L_), l(l_), k(k_) {}
    virtual bool isActive(const lf::mesh::Entity &edge) {
        return bd_flags(edge);
    }
    
    elem_mat_t Eval(const lf::mesh::Entity &edge);
private:
    lf::mesh::utils::CodimMeshDataSet<bool> &bd_flags;
    size_type L;
    size_type l;
    double k;
};
