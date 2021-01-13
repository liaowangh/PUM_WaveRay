#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "PUM_EdgeVector.h"
#include "../utils/triangle_integration.h"

using namespace std::complex_literals;

PUM_EdgeVec::EdgeVec_t PUM_EdgeVec::Eval(const lf::mesh::Entity &edge) {
    const lf::base::RefEl ref_el{edge.RefEl()};
    LF_ASSERT_MSG(ref_el == lf::base::RefEl::kSegment(),"Edge must be of segment type");
    size_type N = (1 << (L + 1 - l));
    EdgeVec_t edge_vec(2 * N);
    edge_vec.setZero();

    for(int t = 0; t < N; ++t) {
        auto new_g = [this, &t, &N](const Eigen::Vector2d& x)->Scalar {
            Eigen::Matrix<std::complex<double>, 2, 1> d;
            double pi = std::acos(-1);
            d << std::cos(2*pi*t/N), std::sin(2*pi*t/N);
            return g(x) * std::exp(-1i * k * d.dot(x));
        };
        lf::mesh::utils::MeshFunctionGlobal mf_g{new_g};
        lf::uscalfe::ScalarLoadEdgeVectorProvider<double, decltype(mf_g), decltype(edge_selector)> 
            Lin_edgeVec_builder(fe_space, mf_g, edge_selector);
        auto edge_vec_tmp = Lin_edgeVec_builder.Eval(edge);
        edge_vec(t) = edge_vec_tmp(0);
        edge_vec(N + t) = edge_vec_tmp(1);
    }
    return edge_vec;
}
