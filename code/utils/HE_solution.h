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
using Scalar = std::complex<double>;
using size_type = unsigned int;
using function_type = std::function<Scalar(Eigen::Vector2d&)>;
using namespace std::complex_literals;

/*
 * Plan waves: u(x) = exp(ik(d1*x(1) + d2*x(2))
 * k is the wave number, and frequency d1^2+d2^2 = 1
 * Overload the operator() to return the value a x.
 * And it should also have a member that is the g.
 */
class plan_wave {
public:
    plan_wave(double k_, double d1_, double d2_): k(k_), d1(d1_), d2(d2_);
    Scalar operator()(const Eigen::Vector2d& x){
        return std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
    }
private:
    double k;
    double d1;
    double d2;
    function_type g;
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
 * d/dx Yv(x) = v/x * Yv(x) - Y_{v+1}(x)
 * d/dx Iv(x) = v/x * Iv(x) + I_{v+1}(x)
 * d/dx Kv(x) = v/x * Kv(x) - K_{v+1}(x)
 */

Scalar cyl_bessel_j_dx(double v, double x);
/*
 * Fundamental solutions: u(x) = i / 4 * H0(ik||x-c||)
 * H0 is the Hankel function of first kind
 * c is the center point, and it should be outside the domain
 *
 * And H0 = J0 + iY0, where
 * J0 is the Bessel function of the first kind with order 0
 * Y0 is the Bessel function of the second kind with order 0
 
 */

class fundamental_sol {
public:
    fundamental_sol(double k_): k(k_);
    Scalar operator()(const Eigen::Vector2d& x);
private:
    double k;
};


/*
 * Spherical wave: u(r, \phi)=J_{|l|}(kr) * exp(i*l*\phi)
 * where r = ||x||, and 0 <= \phi <= 2\pi, l is an integer
 * and J is the Bessel functions of first find
 *
 */
class Spherical_wave{
    Spherical_wave(double k_, int l_);
    Scalar operator()(const Eigen::Vector2d& x);
private:
    double k;
    int l;
    function_type g;
};
