#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "GMRES.h"

using namespace std::complex_literals;
using Scalar = std::complex<double>;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using SpMat_t = Eigen::SparseMatrix<Scalar>;

int main() {
    int N = 15;
    Mat_t A = Mat_t::Random(N, N);
    Mat_t x = Vec_t::Random(N);
    Vec_t b = A * x;

    Vec_t v = Vec_t::Random(N);
    int m = 50;

    Vec_t init;
    init = v;
    gmres(A, b, init, m, 0.0000001, true);
    std::cout << std::left;
    std::cout << std::setw(5) << "m" << std::setw(15) << "||em||" 
              << std::setw(15) << "||rm||" << std::endl;
    for(int i = 1; i <= m; ++i) {
        init = v;
        gmres(A, b, init, i, 1);
        std::cout << std::setw(5) << i << std::setw(15) << (init-x).norm() 
              << std::setw(15) << (b - A*init).norm() / (b-A*v).norm() << std::endl;
    }
}