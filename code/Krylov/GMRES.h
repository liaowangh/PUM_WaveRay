#pragma once

#include <iostream>
#include <iomanip>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std::complex_literals;
using Scalar = std::complex<double>;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using SpMat_t = Eigen::SparseMatrix<Scalar>;
using triplet_t = Eigen::Triplet<Scalar>;

/*
 * use the GMRES to solve the equations Ax=b
 * argument x is the initial value
 * In every iteration, we need to solve an least square problem: y = argmin ||beta - Hm y||
 * Hm is a Hessenberg matrix with size (m+1, m)
 */


/*
 * Solve for linear equations Ux = b
 * where U is a upper triangle matrix
 * we have uii * xi + ui, {i+1} x_{i+1} + ... + uin * xn = bi
 */
Vec_t solve_upper_triangle(const Mat_t& U, const Vec_t& b) {
    int n = b.size();
    Vec_t x = Vec_t(n);
    for(int i = n - 1; i >= 0; --i) {
        Scalar tmp = b(i);
        for(int j = i + 1; j < n; ++j) {
            tmp -= U(i,j) * x(j);
        }
        x(i) = tmp / U(i, i);
    }
    return x;
}

std::pair<Scalar, Scalar> givens_rotation(Scalar rho, Scalar sigma) {
    if(rho == 0.0) {
        return {0.0, 1.0};
    } else {
        double tmp = std::abs(rho * rho) + std::abs(sigma * sigma);
        Scalar c = std::abs(rho) / (std::sqrt(tmp));
        Scalar s = c * sigma / rho;
        return {c, s};
    }
}

void applay_givens_roataion(Mat_t& H, Vec_t& cs, Vec_t& sn, int j) {
    for(int i = 0; i < j; ++i) {
        Scalar tmp = cs(i) * H(i, j) + std::conj(sn(i)) * H(i+1, j);
        H(i+1, j) = -sn(i) * H(i, j) + cs(i) * H(i+1, j);
        H(i, j) = tmp;
    }
    auto cs_pair = givens_rotation(H(j, j), H(j+1, j));
    cs(j) = cs_pair.first;
    sn(j) = cs_pair.second;
    H(j, j) = cs(j) * H(j,j) + std::conj(sn(j)) * H(j+1, j);
    H(j+1, j) = 0.0;
}

template <typename Mat_type>
void gmres(Mat_type& A, Vec_t& b, Vec_t& x, int max_iterations, double threshold, bool verbose = false) {
    int n = A.rows();
    int m = max_iterations;
    Vec_t r = b - A * x; // initial residual
    double r_norm = r.norm();
    double error = r_norm / r_norm;
    std::vector<double> e;
    e.push_back(error);

    // For givens roatation
    Vec_t sn = Vec_t::Zero(m), cs = Vec_t::Zero(m);

    // beta is initialized to r_norm*e1
    Vec_t beta = Vec_t::Zero(m+1);
    beta(0) = r_norm;
  
    Mat_t V(n, m+1);  // to store the orthonormal bases 
    Mat_t H = Mat_t::Zero(m+1, m);
    V.col(0) = r / r_norm;

    int j;
    if(verbose) {
        std::cout << std::left;
        std::cout << std::setw(5) << "m" << std::setw(15) << "||rm||" << std::endl;
    }
    for(j = 1; j <= m; ++j) {
        // arnoldi process
        Vec_t wj = A * V.col(j-1);
        for(int i = 0; i < j; ++i) {
            H(i, j-1) = wj.dot(V.col(i));
            wj = wj - H(i, j-1) * V.col(i);
        }
        H(j, j-1) = wj.norm();
        V.col(j) = wj / H(j, j-1);
        
        // eliminate the last element in H jth row and update the rotation matrix
        applay_givens_roataion(H, cs, sn, j-1);
       
        // update the residual vector
        beta(j)  = -sn(j-1) * beta(j-1);
        beta(j-1) = cs(j-1) * beta(j-1);
        
        error = std::abs(beta(j)) / r_norm;
        e.push_back(error);
        if(verbose) {
            Vec_t y_tmp = solve_upper_triangle(H.topLeftCorner(j, j), beta.head(j));
            Vec_t x_tmp = x + V.leftCols(j) * y_tmp;
            std::cout << std::setw(5) << j << std::setw(15) << error 
                      << std::setw(15) << (b-A*x_tmp).norm() / r_norm << std::endl; 
        }
        if(error <= threshold) {
            if(verbose) {
                std::cout << "GMRES ends after " << j << " iterations." << std::endl;
            }
            break;
        }
    }
    // calculate the result
    if(j == m + 1) {
        --j;
    }
    Vec_t y = solve_upper_triangle(H.topLeftCorner(j, j), beta.head(j));
    x = x + V.leftCols(j) * y;
}





