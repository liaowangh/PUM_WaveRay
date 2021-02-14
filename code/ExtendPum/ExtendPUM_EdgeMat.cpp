#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "ExtendPUM_EdgeMat.h"
#include "../utils/triangle_integration.h"

using namespace std::complex_literals;

// gamma * u * v.conj = gamma * exp(ik(d1-d2) x) * bi * bj
ExtendPUM_EdgeMat::Mat_t ExtendPUM_EdgeMat::Eval(const lf::mesh::Entity& edge) {
    const lf::base::RefEl ref_el{edge.RefEl()};
    LF_ASSERT_MSG(ref_el == lf::base::RefEl::kSegment(),"Edge must be of segment type");
    Mat_t edge_mat(2 * (N_+1), 2 * (N_+1));

    double pi = std::acos(-1.);
    Eigen::Vector2d d1, d2;
    for(int t1 = 0; t1 <= N_; ++t1) {
        if(t1 == 0) {
            d1 << 0.0, 0.0;
        } else {
            d1 << std::cos(2*pi*(t1-1)/N_), std::sin(2*pi*(t1-1)/N_);
        }
        
        for(int t2 = 0; t2 <= N_; ++t2) {
            // edge_mat(i,j) = \int_e gamma * exp(ikdj x) * bj * exp(-ikdi x) * bi
            if(t2 == 0){
                d2 << 0.0, 0.0;
            } else {
                d2 << std::cos(2*pi*(t2-1)/N_), std::sin(2*pi*(t2-1)/N_);
            }

            auto new_gamma = [this, &d1, &d2](const Eigen::Vector2d& x)->Scalar{
                return gamma_ * std::exp(1i * k_ * (d2-d1).dot(x));
            };
            lf::mesh::utils::MeshFunctionGlobal mf_gamma{new_gamma};
            lf::uscalfe::MassEdgeMatrixProvider<double, decltype(mf_gamma), decltype(edge_selector_)> 
                edgeMat_builder(fe_space_, mf_gamma, lf::quad::make_QuadRule(ref_el, degree_), edge_selector_);
            const auto edge_mat_tmp = edgeMat_builder.Eval(edge);
            edge_mat(t1, t2) = edge_mat_tmp(0, 0);
            edge_mat(t1, t2 + N_+1) = edge_mat_tmp(0, 1);
            edge_mat(t1 + N_+1, t2) = edge_mat_tmp(1, 0);
            edge_mat(t1 + N_+1, t2 + N_+1) = edge_mat_tmp(1, 1);
        }
    }
    return edge_mat;
}