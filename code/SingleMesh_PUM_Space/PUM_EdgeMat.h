#pragma once

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>


/*
 * (Local) bilinear form (u,v) -> \int_e gamma * u * v.conj dS
 * where gamma is a complex number, 
 * u = bi(x) * exp(ikd1 x), v.conj = bj(x) * exp(-ikd2 x)
 * and gamma * u * v.conj = gamma * exp(ik(d1-d2) x) * bi * bj
 * we can make use of lf::uscalfe::MassEdgeMatrixProvider
 */
class PUM_EdgeMat {
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    
    PUM_EdgeMat(std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_, 
        lf::mesh::utils::CodimMeshDataSet<bool> &edge_selector_, size_type N_, double k_, Scalar gamma_): 
        fe_space(fe_space_), edge_selector(edge_selector_), N(N_), k(k_), gamma(gamma_) {}

    bool isActive(const lf::mesh::Entity &edge) {
        return edge_selector(edge);
    }
    
    Mat_t Eval(const lf::mesh::Entity &edge);

private:
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space;
    lf::mesh::utils::CodimMeshDataSet<bool> edge_selector;
    size_type N; // number of planar waves
    double k;
    Scalar gamma;
};
