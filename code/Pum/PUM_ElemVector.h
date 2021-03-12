#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

/*
 * (local) Linear form: l(v) = \int_K f v.conjdS(x),
 * v = bi * exp(ikdt x), v.conj = bi * exp(-ikdt x)
 */
class PUM_ElemVec{
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    
    PUM_ElemVec(std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
        size_type N, double k, FHandle_t f, int degree=20): 
        fe_space_(fe_space), N_(N), k_(k), f_(f), degree_(degree){}
    
    bool isActive(const lf::mesh::Entity& cell) { return true; }
    
    Vec_t Eval(const lf::mesh::Entity &cell) {
        const lf::base::RefEl ref_el{cell.RefEl()};
        LF_ASSERT_MSG(ref_el == lf::base::RefEl::kTria(),
                    "Cell must be of triangle type");
        Vec_t elem_vec(3 * N_);

        std::map<lf::base::RefEl, lf::quad::QuadRule> quad_rules;
        quad_rules[lf::base::RefEl::kTria()] = 
            lf::quad::make_QuadRule(lf::base::RefEl::kTria(), degree_);
        quad_rules[lf::base::RefEl::kQuad()] = 
            lf::quad::make_QuadRule(lf::base::RefEl::kQuad(), degree_);

        for(int t = 0; t < N_; ++t) {
            auto f_new = [this, &t](const Eigen::Vector2d& x)->Scalar {
                Eigen::Matrix<double, 2, 1> d;
                double pi = std::acos(-1);
                d << std::cos(2*pi*t/N_), std::sin(2*pi*t/N_);
                return f_(x) * std::exp(-1i * k_ * d.dot(x));
            };
            lf::mesh::utils::MeshFunctionGlobal mf_f{f_new};
            lf::uscalfe::ScalarLoadElementVectorProvider<double, decltype(mf_f)>
                ElemVec_builder(fe_space_, mf_f, quad_rules);
            Vec_t vec_tmp = ElemVec_builder.Eval(cell);
            for(int i = 0; i < 3; ++i) {
                elem_vec(i*N_+t) = vec_tmp(i);
            }
        }
        return elem_vec;
    }
private:
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_;
    size_type N_; // number of planner waves
    double k_;
    FHandle_t f_;
    int degree_;
};

