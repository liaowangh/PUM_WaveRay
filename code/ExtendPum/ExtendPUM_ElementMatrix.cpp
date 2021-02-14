#include <cmath>
#include <vector>

#include "ExtendPUM_ElementMatrix.h"
#include "../utils/utils.h"
#include "../utils/triangle_integration.h"

using namespace std::complex_literals;

// a(u,v) = alpha * (grad u, grad v) + gamma * (u, v)
ExtendPUM_ElementMatrix::Mat_t ExtendPUM_ElementMatrix::Eval(const lf::mesh::Entity& cell) {
    const lf::base::RefEl ref_el{cell.RefEl()};
    LF_ASSERT_MSG(ref_el == lf::base::RefEl::kTria(),
                  "Cell must be of triangle type");
    Mat_t elem_mat(3 * (N_+1), 3 * (N_+1));

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
    
    for(int i = 0; i < 3*(N_+1); ++i){
        // ci = b_i1 * e_t1
        int i1 = i / (N_+1);
        int t1 = i % (N_+1);
        for(int j = 0; j < 3*(N_+1); ++j) {
            // cj = b_j2 * e_t2
            // elem_mat(i,j) = aK(cj, ci)
            int j2 = j / (N_+1);
            int t2 = j % (N_+1);
            auto f = [this,&X,&i1,&j2,&t1,&t2](const Eigen::Vector2d& x)->Scalar {
                Eigen::Vector2d di, dj, betai, betaj; 
                double pi = std::acos(-1);
                if(t1 == 0) {
                    di << 0.0, 0.0;
                } else {
                    di << std::cos(2*pi*(t1-1)/N_), std::sin(2*pi*(t1-1)/N_);
                }
                if(t2 == 0) {
                    dj << 0.0, 0.0;
                } else {
                    dj << std::cos(2*pi*(t2-1)/N_), std::sin(2*pi*(t2-1)/N_);
                }
                betai << X(1, i1), X(2, i1);
                betaj << X(1, j2), X(2, j2);
                double lambdai = X(0,i1) + betai.dot(x);
                double lambdaj = X(0,j2) + betaj.dot(x);

                auto gradci = std::exp(1i*k_*di.dot(x)) * (betai + 1i*k_*lambdai*di);
                auto gradcj = std::exp(1i*k_*dj.dot(x)) * (betaj + 1i*k_*lambdaj*dj);
                auto val_ci = lambdai * std::exp(1i*k_*di.dot(x));
                auto val_cj = lambdaj * std::exp(1i*k_*dj.dot(x));

                // !!!! in Eigen, u.dot(v) return the hermitian dot product (equals to u.adjoint()*v)
                return alpha_ * gradci.dot(gradcj) + gamma_ * val_cj * std::conj(val_ci);
            }; 
            elem_mat(i, j) = LocalIntegral(cell, degree_, f);
        }
    }
    return elem_mat;
}