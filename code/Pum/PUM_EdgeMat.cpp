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

#include "PUM_EdgeMat.h"
#include "../utils/triangle_integration.h"

using namespace std::complex_literals;

// gamma * u * v.conj = gamma * exp(ik(d1-d2) x) * bi * bj
PUM_EdgeMat::Mat_t PUM_EdgeMat::Eval(const lf::mesh::Entity& edge) {
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
                edgeMat_builder(fe_space_, mf_gamma, lf::quad::make_QuadRule(ref_el, degree_), edge_selector);
            const auto edge_mat_tmp = edgeMat_builder.Eval(edge);
            edge_mat(t1, t2) = edge_mat_tmp(0, 0);
            edge_mat(t1, t2 + N_) = edge_mat_tmp(0, 1);
            edge_mat(t1 + N_, t2) = edge_mat_tmp(1, 0);
            edge_mat(t1 + N_, t2 + N_) = edge_mat_tmp(1, 1);
        }
    }
    return edge_mat;
}


/*
EdgeMatProvider::elem_mat_t EdgeMatProvider::Eval(const lf::mesh::Entity &edge) {
    LF_VERIFY_MSG(edge.RefEl() == lf::base::RefEl::kSegment(),
                  "Unsupported edge type " << edge.RefEl());
    // obtain endpoint coordinates of the triangle in a 2x2 matrix
    const auto endpoints = lf::geometry::Corners(*(edge.Geometry()));
    
    const double edge_length = (endpoints.col(1) - endpoints.col(0)).norm();
    
    size_type N = (1 << (L + 1 - 1));
    elem_mat_t edge_mat(2*N, 2*N);
    
    Eigen::Vector2d d1, d2, da;
    da = endpoints.col(1) - endpoints.col(0);
    
    for(int i1 = 0; i1 < 2; i1++){
        for(int j1 = 0; j1 < N; ++j1) {
            d1 << std::cos(2*M_PI*j1 / N), std::sin(2*M_PI*j1 / N);
            for(int i2 = 0; i2 < 2; i2++) {
                for(int j2 = 0; j2 < N; ++j2) {
                    d2 << std::cos(2*M_PI*j2 / N), std::sin(2*M_PI*j2 / N);
                    
                    // \int_0^1 (a*t^2+b*t+c) exp(dt) dt
                    double a, b, c;
                    if(i1 - i2) {
                        a = -1 / (edge_length * edge_length);
                        b = 1 / edge_length;
                        c = 0;
                    } else if(i1 == 0) {
                        a = 1 / (edge_length * edge_length);
                        b = 0;
                        c = 0;
                    } else {
                        a = 1 / (edge_length * edge_length);
                        b = -2 / edge_length;
                        c = 1;
                    }
                    mat_scalar d, coeff, tmp;
                    coeff = 1i*k*std::exp(1i*k*(d1+d2).dot(endpoints.col(0)));
                    d = 1i * k * (d1 + d2).dot(da);
                    tmp = a * int_x2exp(0, 1, d) + b * int_xexp(0, 1, d) + c * int_exp(0, 1, d);
                    tmp *= coeff;
                    edge_mat(i1*N+j1, i2*N+j2) = tmp;
                }
            }
        }
    }
    return edge_mat;
}
*/