#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "HE_solution.h"

using namespace std::complex_literals;

/*
 * Plan waves
 */
FHandle_t plan_wave::get_fun() {
    return [this](const coordinate_t& x)-> Scalar {
        return std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
    };
}

FunGradient_t plan_wave::get_gradient() {
    return [this](const coordinate_t& x) {
        auto tmp = 1i * k * std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
        Eigen::Matrix<Scalar, 2, 1> res;
        res << tmp * d1, tmp * d2;
        return res;
    };
}

FHandle_t plan_wave::boundary_g() {
    auto g = [this](const coordinate_t& x) -> Scalar {
        double x1 = x(0), y1 = x(1);
        Scalar res = 1i * k * std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
        if(y1 == 0 && x1 <= 1 && x1 >= 0) {
            // (0,-1)
            res *= (-d2 - 1);
        } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
            // (1,0)
            res *= (d1 - 1);
        } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
            //(0, 1)
            res *= (d2 - 1);
        } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
            // (-1,0)
            res *= (-d1 - 1);
        } 
        // else if(x1 + y1 == 1) {
        //     // \sqrt(2) * (1,1)
        //     res *= (std::sqrt(2)*(d1 + d2)/2 - 1);
        // }
        return res;
    };
    return g;
}

// /*
//  * Bessel functions and Hankel functions
//  */
// // Jv(ix) = exp(iv\pi/2)*Iv(x), x \in R
// Scalar bessel_j_ix(double v, double x){
//     return std::exp(1i * v * std::acos(-1) / 2.) * std::cyl_bessel_i(v, x);
// }

// // Yv(ix) = exp(i(v+1)\pi/2)*Iv(x) - 2exp(-iv\pi/2)/pi*Kv(x)
// Scalar bessel_y_ix(double v, double x){
//     return 
//         std::exp(1i*(v+1)* std::acos(-1) / 2.) * std::cyl_bessel_i(v, x) - 
//         std::exp(-1i * v * std::acos(-1) / 2.) * std::cyl_bessel_k(v, x) * 2. / std::acos(-1);
// }

// Scalar cyl_bessel_j_dx(double v, double x) {
//     return x == 0 ? 0 : v * std::cyl_bessel_j(v,x) / x - std::cyl_bessel_j(v+1, x);
// }

// Scalar cyl_neumann_dx(double v, double x) {
//     return v * std::cyl_neumann(v, x) / x - std::cyl_neumann(v+1, x);
// }

// Scalar hankel_1(double v, double x) {
//     return std::cyl_bessel_j(v, x) + 1i * std::cyl_neumann(v, x);
// }

// Scalar hankel_1_dx(double v, double x) {
//     return cyl_bessel_j_dx(v, x) + 1i * cyl_neumann_dx(v, x);
// }

// // Hankel function taking pure imaginary argument
// Scalar hankel_1_ix(double v, double x){
//     return bessel_j_ix(v, x) + 1i * bessel_y_ix(v, x);
// }
// // Derivative of Hankel function taking pure imaginary argument
// Scalar hankel_1_dx_ix(double v, double x) {
//     return v * bessel_j_ix(v, x) / x - bessel_j_ix(v+1, x) + 
//             1i * (v * bessel_y_ix(v, x) / x - bessel_y_ix(v+1, x));
// }

// /*
//  * Fundamental solution
//  */
// //Scalar fundamental_sol::operator()(const coordinate_t& x) {
// //    return hankel_1(k * (x - c).norm());
// //}

// FHandle_t fundamental_sol::get_fun() {
//     return [this](const coordinate_t& x) -> Scalar {
//         return hankel_1(0, k * (x - c).norm());
//     };
// }

// // grad u = H' * k * (x-c)/ ||x-c||
// FunGradient_t fundamental_sol::get_gradient() {
//     return [this](const coordinate_t& x) {
//         double x1 = x(0), y1 = x(1);
//         double c1 = c(0), c2 = c(1);
        
//         double r = (x - c).norm();

//         Scalar u = hankel_1(0, k * r);
//         Scalar dudx = hankel_1_dx(0, k * r) * k * (x1 - c1) / r;
//         Scalar dudy = hankel_1_dx(0, k * r) * k * (y1 - c2) / r;
//         return (Eigen::Matrix<Scalar, 2, 1>() << dudx, dudy).finished();
//     };
// }

// FHandle_t fundamental_sol::boundary_g() {
//     // u(x) = H0(k*||x-c||)
//     // grad u = H' * k * (x-c)/ ||x-c||
//     auto g = [this](const coordinate_t& x) -> Scalar {
//         double x1 = x(0), y1 = x(1);
//         double c1 = c(0), c2 = c(1);
        
//         double r = (x - c).norm();

//         Scalar u = hankel_1(0, k * r);
//         Scalar dudx = hankel_1_dx(0, k * r) * k * (x1 - c1) / r;
//         Scalar dudy = hankel_1_dx(0, k * r) * k * (y1 - c2) / r;

//         Scalar res = -1i * k * u;
//         if(y1 == 0 && x1 <= 1 && x1 >= 0) {
//             // n =  (0, -1)
//             res += -1. * dudy;
//         } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
//             // n = (1, 0)
//             res += dudx;
//         } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
//             // n = (0, 1)
//             res += dudy;
//         } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
//             // n = (-1, 0)
//             res += -1. * dudx;
//         }
//         return res;
//     };
//     return g;
// }

// /*
//  * Spherical waves
//  */

// //Scalar Spherical_wave::operator()(const coordinate_t& x){
// //    double r = std::sqrt(x(0)^2 + x(1)^2);
// //    double phi = std::atan2(x(1), x(0));
// //    return std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * l * phi);
// //}

// FHandle_t Spherical_wave::get_fun() {
//     auto u = [this](const coordinate_t& x) -> Scalar {
//         double r = x.norm();
//         double phi = std::atan2(x(1), x(0));
//         return std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * phi * l);
//     };
//     return u;
// }

// FunGradient_t Spherical_wave::get_gradient() {
//     return [this](const coordinate_t& x) {
//         double x1 = x(0), y1 = x(1);
//         double r = x.norm();
//         double sin_ = y1 / r, cos_ = x1 / r;
//         double phi = std::atan2(y1, x1);
        
//         Scalar u = std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * phi * l);
//         Scalar dudx = (cos_ * k * cyl_bessel_j_dx(std::abs(l), k*r) -
//                        1i * sin_ * l * std::cyl_bessel_j(std::abs(l), k*r) / r) * std::exp(1i*phi*l);
//         Scalar dudy = (sin_ * k * cyl_bessel_j_dx(std::abs(l), k*r) +
//                        1i * cos_ * l * std::cyl_bessel_j(std::abs(l), k*r) / r) * std::exp(1i*phi*l);
//         return (Eigen::Matrix<Scalar, 2, 1>() << dudx, dudy).finished();
//     };
// }

// FHandle_t Spherical_wave::boundary_g() {
//     auto g = [this](const coordinate_t& x) -> Scalar {
//         double x1 = x(0), y1 = x(1);
//         if(x1 == 0 && y1 == 0) {
//             return 0;
//         }
//         double r = x.norm();
//         double sin_ = y1 / r, cos_ = x1 / r;
//         double phi = std::atan2(y1, x1);
        
//         Scalar u = std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * phi * l);
//         Scalar dudx = (cos_ * k * cyl_bessel_j_dx(std::abs(l), k*r) -
//                        1i * sin_ * l * std::cyl_bessel_j(std::abs(l), k*r) / r) * std::exp(1i*phi*l);
//         Scalar dudy = (sin_ * k * cyl_bessel_j_dx(std::abs(l), k*r) +
//                        1i * cos_ * l * std::cyl_bessel_j(std::abs(l), k*r) / r) * std::exp(1i*phi*l);
//         Scalar res = -1i * k * u;
//         if(y1 == 0 && x1 <= 1 && x1 >= 0) {
//             // n =  (0, -1)
//             res += -1. * dudy;
//         } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
//             // n = (1, 0)
//             res += dudx;
//         } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
//             // n = (0, 1)
//             res += dudy;
//         } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
//             // n = (-1, 0)
//             res += -1. * dudx;
//         }
//         return res;
//     };
//     return g;
// }

// /*
//  * Harmonic function 
//  * Solution for Laplace equation (k = 0 in Helmholtz equation)
//  * We take the function u(x,y)=exp(x+iy)
//  */
// FHandle_t Harmonic_fun::get_fun() {
//     return [](const coordinate_t& x) -> Scalar {
//         return std::exp(x(0)+1i*x(1));
//     };
// }

// FunGradient_t Harmonic_fun::get_gradient() {
//     return [](const coordinate_t& x) {
//         double x1 = x(0), y1 = x(1);

//         Scalar u = std::exp(x1+1i*y1);
//         Scalar dudx = u;
//         Scalar dudy = 1i * u;
//         return (Eigen::Matrix<Scalar, 2, 1>() << dudx, dudy).finished();
//     };
// }

// FHandle_t Harmonic_fun::boundary_g() {
//     auto g = [](const coordinate_t& x) -> Scalar {
//         double x1 = x(0), y1 = x(1);

//         Scalar u = std::exp(x1+1i*y1);
//         Scalar dudx = u;
//         Scalar dudy = 1i * u;
//         Scalar res = 0.;
//         if(y1 == 0 && x1 <= 1 && x1 >= 0) {
//             // n =  (0, -1)
//             res += -1. * dudy;
//         } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
//             // n = (1, 0)
//             res += dudx;
//         } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
//             // n = (0, 1)
//             res += dudy;
//         } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
//             // n = (-1, 0)
//             res += -1. * dudx;
//         }
//         return res;
//     };
//     return g;
// }