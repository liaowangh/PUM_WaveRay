#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

/*
 * Manufacture solutions for Helmholtz equation
 *  \Laplace u + k^2 * u = 0
 *  \partial u / \partial n - iku = g on \Gamma_R
 *  u = h on \Gamma_D
 
 * The domain is [0,1] x [0,1] \ [0.375, 0.375] x [0.375, 0.375]
 * And \Gamma_R is the boundary of square [0,1] x [0,1]
 *     \Gamma_D is the boundary of square [0.375, 0.375] x [0.375, 0.375]
 */
using coordinate_t = Eigen::Vector2d;
using Scalar = std::complex<double>;
using size_type = unsigned int;
using FHandle_t = std::function<Scalar(const coordinate_t&)>;
using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

using namespace std::complex_literals;

class HE_sol {
public:
    HE_sol(Scalar wave_num): k(wave_num){};
    
    //virtual Scalar operator()(const coordinate_t&) = 0;
    virtual FHandle_t get_fun() = 0;
    virtual FunGradient_t get_gradient() = 0;
    virtual FHandle_t boundary_g() = 0;
    virtual ~HE_sol() = default;

public:
    Scalar k;
};


/*
 * Plan waves: u(x) = exp(ik(d1*x(0) + d2*x(1))
 * k is the wave number, and frequency d1^2+d2^2 = 1
 * 
 * grad u = ik * u(x) * [d1, d1]
 */
class plan_wave: public HE_sol {
public:
    plan_wave(Scalar k_, double d1_, double d2_): HE_sol(k_), d1(d1_), d2(d2_){}
//    Scalar operator()(const coordinate_t& x) override;
    FHandle_t get_fun() override;
    FunGradient_t get_gradient() override;
    FHandle_t boundary_g() override;
private:
    double d1;
    double d2;
};

/*
 * Bessel functions in C++, (x >= 0)
 * Jv(x) = std::cyl_bessel_j(v, x)
 * Yv(x) = std::cyl_neumann(v, x)
 * Modified Bessel functions of first and second kind:
 * Iv(x) = std::cyl_bessel_i(v, x):
 * Kv(x) = std::cyl_bessel_k(v, x)
 *
 * Derivatives: https://www.boost.org/doc/libs/1_75_0/libs/math/doc/html/math_toolkit/bessel/bessel_over.html
 * d/dx Jv(x) = v/x * Jv(x) - J_{v+1}(x)
 *            = (J_{v-1}(x) - J_{v+1}(x)) / 2
 * d/dx Yv(x) = v/x * Yv(x) - Y_{v+1}(x)
 *            = (Y_{v-1}(x) - Y_{v+1}(x)) / 2
 * d/dx Iv(x) = v/x * Iv(x) + I_{v+1}(x)
 * d/dx Kv(x) = v/x * Kv(x) - K_{v+1}(x)
 */

// // Jv(ix) = exp(iv\pi/2)*Iv(x), x \in R
// Scalar bessel_j_ix(double v, double x);
// // Yv(ix) = exp(i(v+1)\pi/2)*Iv(x) - 2exp(-iv\pi/2)/pi*Kv(x)
// Scalar bessel_y_ix(double v, double x);

// // derivative of Jv at x
// Scalar cyl_bessel_j_dx(double v, double x);
// // derivative of Yv at x
// Scalar cyl_neumann_dx(double v, double x);
// // Hankel function of first kind: Hv = Jv + i Yv
// Scalar hankel_1(double v, double x);
// // derivative of Hv
// Scalar hankel_1_dx(double v, double x);
// // Hankel function taking pure imaginary argument
// Scalar hankel_1_ix(double v, double x);
// // Derivative of Hankel function taking pure imaginary argument
// Scalar hankel_1_dx_ix(double v, double x);

// /*
//  * Fundamental solutions: u(x) = H0(k||x-c||) (not sure)
//  * H0 is the Hankel function of first kind
//  * c is the center point, and it should be outside the domain
//  *
//  * And H0 = J0 + iY0, where
//  * J0 is the Bessel function of the first kind with order 0
//  * Y0 is the Bessel function of the second kind with order 0
//  */
// class fundamental_sol: public HE_sol {
// public:
//     fundamental_sol(Scalar wave_num, coordinate_t c_): HE_sol(wave_num), c(c_){}
//     FHandle_t get_fun() override;
//     FunGradient_t get_gradient() override;
//     FHandle_t boundary_g() override;
// private:
//     coordinate_t c;
// };


// /*
//  * Spherical wave: u(r, \phi)=J_{|l|}(kr) * exp(i*l*\phi)
//  * where r = ||x||, and 0 <= \phi <= 2\pi, l is an integer
//  * and J is the Bessel functions of first find
//  *
//  */
// class Spherical_wave: public HE_sol {
// public:
//     Spherical_wave(Scalar wave_num, double l_): HE_sol(wave_num), l(l_){}
//     FHandle_t get_fun() override;
//     FunGradient_t get_gradient() override;
//     FHandle_t boundary_g() override;
// private:
//     double l;
// };

// /*
//  * Manufacture solutions for Laplace equation
//  *  \Laplace u = 0 (special case for Helmholtz equation)
//  *  \partial u / \partial n - iku = g on \Gamma_R
//  *  u = h on \Gamma_D
 
//  * The domain is [0,1] x [0,1] \ [0.375, 0.375] x [0.375, 0.375]
//  * And \Gamma_R is the boundary of square [0,1] x [0,1]
//  *     \Gamma_D is the boundary of square [0.375, 0.375] x [0.375, 0.375]
//  */
//  class Harmonic_fun {
//  public:
//     Harmonic_fun(){};
//     FHandle_t get_fun();
//     FunGradient_t get_gradient();
//     FHandle_t boundary_g();
//  };

