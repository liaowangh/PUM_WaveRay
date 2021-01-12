#include "PUM_ElemVector.h"

using namespace std::complex_literals;

PUM_ElemVec::ElemVec_t PUM_ElemVec::eval(const lf::mesh::Entity& cell) {
    const lf::base::RefEl ref_el{edge.RefEl()};
    LF_ASSERT_MSG(ref_el == lf::base::RefEl::kTria(),
                  "Cell must be of triangle type");
    size_type N = (1 << (L + 1 - l));
    ElemVec_t elemVec(3 * N);

    for(int t = 0; t < N; ++t) {
        auto f_new = [this, &t, &N](const Eigen::Vector2d& x)->Scalar {
            Eigen::Matrix<std::complex<double>, 2, 1> d;
            double pi = std::acos(-1);
            d << std::cos(2*pi*t/N), std::sin(2*pi*t/N);
            return f(x) * std::exp(-1i * k * d.dot(x));
        };
        lf::uscalfe::ScalarLoadElementVectorProvider<double, decltype(f_new)>
        ElemVec_builder(fe_space, f_new);
        elemVec_tmp = ElemVec_builder.eval(cell);
        for(int i = 0; i < 3; ++i) {
            elemVec(i*N+t) = elemVec_tmp(i);
        }
    }
    return elemVec;
}