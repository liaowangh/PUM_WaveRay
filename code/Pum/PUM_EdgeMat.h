#pragma once

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>


/*
 * (Local) bilinear form (u,v) -> \int_e gamma * (u,v) dS
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
    
    PUM_EdgeMat(std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space, 
        lf::mesh::utils::CodimMeshDataSet<bool> &edge_selector, 
        size_type N, Scalar k, Scalar gamma, int degree=20): 
        fe_space_(fe_space), edge_selector_(edge_selector), N_(N), k_(k), gamma_(gamma), degree_(degree) {}

    bool isActive(const lf::mesh::Entity &edge) {
        return edge_selector_(edge);
    }
    
    Mat_t Eval(const lf::mesh::Entity &edge){
        const lf::base::RefEl ref_el{edge.RefEl()};
        LF_ASSERT_MSG(ref_el == lf::base::RefEl::kSegment(),"Edge must be of segment type");
        Mat_t edge_mat(2 * N_, 2 * N_);

        double pi = std::acos(-1.);
        Eigen::Matrix<Scalar, 2, 1> d1, d2;
        for(int t1 = 0; t1 < N_; ++t1) {
            d1 << std::cos(2*pi*t1/N_), std::sin(2*pi*t1/N_);
            
            for(int t2 = 0; t2 < N_; ++t2) {
                // edge_mat(i,j) = \int_e gamma * exp(ikdj x) * bj * exp(-ikdi x) * bi
                d2 << std::cos(2*pi*t2/N_), std::sin(2*pi*t2/N_);

                auto new_gamma = [this, &d1, &d2](const Eigen::Vector2d& x)->Scalar{
                    return gamma_ * std::exp(1i * k_ * (d2-d1).dot(x));
                };
                lf::mesh::utils::MeshFunctionGlobal mf_gamma{new_gamma};
                lf::uscalfe::MassEdgeMatrixProvider<double, decltype(mf_gamma), decltype(edge_selector_)> 
                    edgeMat_builder(fe_space_, mf_gamma, lf::quad::make_QuadRule(ref_el, degree_), edge_selector_);
                const auto edge_mat_tmp = edgeMat_builder.Eval(edge);
                edge_mat(t1, t2) = edge_mat_tmp(0, 0);
                edge_mat(t1, t2 + N_) = edge_mat_tmp(0, 1);
                edge_mat(t1 + N_, t2) = edge_mat_tmp(1, 0);
                edge_mat(t1 + N_, t2 + N_) = edge_mat_tmp(1, 1);
            }
        }
        return edge_mat;
    };

private:
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_;
    lf::mesh::utils::CodimMeshDataSet<bool> edge_selector_;
    size_type N_; // number of planar waves
    Scalar k_;
    Scalar gamma_;
    int degree_;
};
