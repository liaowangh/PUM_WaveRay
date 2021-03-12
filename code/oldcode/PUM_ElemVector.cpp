#include <map>

#include "PUM_ElemVector.h"
#include "../utils/utils.h"

using namespace std::complex_literals;
/*
// use numerical quadrature LocalIntegral
PUM_ElemVec::Vec_t PUM_ElemVec::Eval(const lf::mesh::Entity& cell) {
       const lf::base::RefEl ref_el{cell.RefEl()};
    LF_ASSERT_MSG(ref_el == lf::base::RefEl::kTria(),
                  "Cell must be of triangle type");
    Vec_t elem_vec(3 * N);

    // Obtain the vertex coordinates of the cell, which completely
    // describe its shape.
    const lf::geometry::Geometry *geo_ptr = cell.Geometry();
    
    // Matrix storing corner coordinates in its columns(2x3 in this case)
    auto vertices = geo_ptr->Global(ref_el.NodeCoords());
    
    // suppose that the barycentric coordinate functions have the form
    // \lambda_i = a + b1*x+b2*y
    // \lambda_i = X(0,i) + X(1,i)*x + X(2,i)*y
    // grad \lambda_i = [X(1,i), X(2,i)]^T
    // grad \lambda_1_2_3 = X.block<2,3>(1,0)
    Eigen::Matrix3d X, tmp;
    tmp.block<3,1>(0,0) = Eigen::Vector3d::Ones();
    tmp.block<3,2>(0,1) = vertices.transpose();
    X = tmp.inverse();
    
    for(int j = 0; j < 3*N; ++j) {
        int i = j / N;
        int t = j % N;
        // integrand: f(x) * bi * exp(-ikdt x)
        auto new_f = [this, &X, &i, &t](const Eigen::Vector2d& x)->Scalar {
            Eigen::Matrix<Scalar, 2, 1> d;
            Eigen::Vector2d beta;
            double pi = std::acos(-1);
            d << std::cos(2*pi*t/N), std::sin(2*pi*t/N);
            beta << X(1, i), X(2, i);
            double lambda = X(0,i) + beta.dot(x);

            return f(x) * lambda * std::exp(-1i*k*d.dot(x));
        };
        elem_vec(j) = LocalIntegral(cell, 10, new_f);
    }
    return elem_vec;
}
*/

// make use of ScalarLoadElementVectorProvider
PUM_ElemVec::Vec_t PUM_ElemVec::Eval(const lf::mesh::Entity& cell) {
    const lf::base::RefEl ref_el{cell.RefEl()};
    LF_ASSERT_MSG(ref_el == lf::base::RefEl::kTria(),
                  "Cell must be of triangle type");
    Vec_t elem_vec(3 * N);

    std::map<lf::base::RefEl, lf::quad::QuadRule> quad_rules;
    quad_rules[lf::base::RefEl::kTria()] = 
        lf::quad::make_QuadRule(lf::base::RefEl::kTria(), 10);
    quad_rules[lf::base::RefEl::kQuad()] = 
        lf::quad::make_QuadRule(lf::base::RefEl::kQuad(), 10);

    for(int t = 0; t < N; ++t) {
        auto f_new = [this, &t](const Eigen::Vector2d& x)->Scalar {
            Eigen::Matrix<double, 2, 1> d;
            double pi = std::acos(-1);
            d << std::cos(2*pi*t/N), std::sin(2*pi*t/N);
            return f(x) * std::exp(-1i * k * d.dot(x));
        };
        lf::mesh::utils::MeshFunctionGlobal mf_f{f_new};
        lf::uscalfe::ScalarLoadElementVectorProvider<double, decltype(mf_f)>
            ElemVec_builder(fe_space, mf_f, quad_rules);
        Vec_t vec_tmp = ElemVec_builder.Eval(cell);
        for(int i = 0; i < 3; ++i) {
            elem_vec(i*N+t) = vec_tmp(i);
        }
    }
    return elem_vec;
}
