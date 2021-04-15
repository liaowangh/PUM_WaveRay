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
 * (Local) linear form: l(v) = \int_e g * v.conj dS(x)
 */
class ExtendPUM_EdgeVec{
public:
    using size_type = unsigned int;
    using Scalar = std::complex<double>;
    using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
    
    ExtendPUM_EdgeVec(std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
        lf::mesh::utils::CodimMeshDataSet<bool> &edge_selector, size_type N, double k, FHandle_t g, int degree = 20): 
        fe_space_(fe_space), edge_selector_(edge_selector), N_(N), k_(k), g_(g), degree_(degree){}
    
    bool isActive(const lf::mesh::Entity& edge) { return edge_selector_(edge); }
    
    Vec_t Eval(const lf::mesh::Entity &edge){
        const lf::base::RefEl ref_el{edge.RefEl()};
        LF_ASSERT_MSG(ref_el == lf::base::RefEl::kSegment(),"Edge must be of segment type");
        Vec_t edge_vec(2 * (N_+1));
        edge_vec.setZero();

        for(int t = 0; t <= N_; ++t) {
            auto new_g = [this, &t](const Eigen::Vector2d& x)->Scalar {
                if(t == 0) {
                    return g_(x);
                } else {
                    Eigen::Vector2d d;
                    double pi = std::acos(-1);
                    d << std::cos(2*pi*(t-1)/N_), std::sin(2*pi*(t-1)/N_);
                    return g_(x) * std::exp(-1i * k_ * d.dot(x));
                }
            };
            lf::mesh::utils::MeshFunctionGlobal mf_g{new_g};
            lf::uscalfe::ScalarLoadEdgeVectorProvider<double, decltype(mf_g), decltype(edge_selector_)> 
                Lin_edgeVec_builder(fe_space_, mf_g, lf::quad::make_QuadRule(ref_el, degree_), edge_selector_);
            auto edge_vec_tmp = Lin_edgeVec_builder.Eval(edge);
            edge_vec(t) = edge_vec_tmp(0);
            edge_vec(N_ + t+1) = edge_vec_tmp(1);
        }
        return edge_vec;
    }
private:
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space_;
    lf::mesh::utils::CodimMeshDataSet<bool> edge_selector_;
    size_type N_; // number of planar waves
    double k_;
    FHandle_t g_;
    int degree_;
};

