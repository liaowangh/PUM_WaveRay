#include <cmath>
#include <functional>

#include <lf/base/base.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "PUM_EdgeVector.h"
#include "../utils/triangle_integration.h"

using namespace std::complex_literals;

PUM_EdgeVec::Vec_t PUM_EdgeVec::Eval(const lf::mesh::Entity &edge) {
    const lf::base::RefEl ref_el{edge.RefEl()};
    LF_ASSERT_MSG(ref_el == lf::base::RefEl::kSegment(),"Edge must be of segment type");
    Vec_t edge_vec(2 * N_);
    edge_vec.setZero();

    for(int t = 0; t < N_; ++t) {
        auto new_g = [this, &t](const Eigen::Vector2d& x)->Scalar {
            Eigen::Matrix<Scalar, 2, 1> d;
            double pi = std::acos(-1);
            d << std::cos(2*pi*t/N_), std::sin(2*pi*t/N_);
            return g_(x) * std::exp(-1i * k_ * d.dot(x));
        };
        lf::mesh::utils::MeshFunctionGlobal mf_g{new_g};
        lf::uscalfe::ScalarLoadEdgeVectorProvider<double, decltype(mf_g), decltype(edge_selector_)> 
            Lin_edgeVec_builder(fe_space_, mf_g, lf::quad::make_QuadRule(ref_el, degree_), edge_selector_);
        auto edge_vec_tmp = Lin_edgeVec_builder.Eval(edge);
        edge_vec(t) = edge_vec_tmp(0);
        edge_vec(N_ + t) = edge_vec_tmp(1);
    }
    return edge_vec;
}

