#include <map>

#include "ExtendPUM_ElemVector.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

// make use of ScalarLoadElementVectorProvider
ExtendPUM_ElemVec::Vec_t ExtendPUM_ElemVec::Eval(const lf::mesh::Entity& cell) {
    const lf::base::RefEl ref_el{cell.RefEl()};
    LF_ASSERT_MSG(ref_el == lf::base::RefEl::kTria(),
                  "Cell must be of triangle type");
    Vec_t elem_vec(3 * (N_+1));

    std::map<lf::base::RefEl, lf::quad::QuadRule> quad_rules;
    quad_rules[lf::base::RefEl::kTria()] = 
        lf::quad::make_QuadRule(lf::base::RefEl::kTria(), degree_);
    quad_rules[lf::base::RefEl::kQuad()] = 
        lf::quad::make_QuadRule(lf::base::RefEl::kQuad(), degree_);

    for(int t = 0; t <= N_; ++t) {
        auto f_new = [this, &t](const Eigen::Vector2d& x)->Scalar {
            if(t == 0) {
                return f_(x);
            } else {
                Eigen::Vector2d d;
                double pi = std::acos(-1);
                d << std::cos(2*pi*(t-1)/N_), std::sin(2*pi*(t-1)/N_);
                return f_(x) * std::exp(-1i * k_ * d.dot(x));
            }
        };
        lf::mesh::utils::MeshFunctionGlobal mf_f{f_new};
        lf::uscalfe::ScalarLoadElementVectorProvider<double, decltype(mf_f)>
            ElemVec_builder(fe_space_, mf_f, quad_rules);
        Vec_t vec_tmp = ElemVec_builder.Eval(cell);
        for(int i = 0; i < 3; ++i) {
            elem_vec(i*(N_+1)+t) = vec_tmp(i);
        }
    }
    return elem_vec;
}
