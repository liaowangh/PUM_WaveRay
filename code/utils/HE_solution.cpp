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
//Scalar plan_wave::operator()(const coordinate_t& x) {
//    return std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
//}

function_type plan_wave::get_fun() {
    return [this](const coordinate_t& x)-> Scalar {
        return std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
    };
}

function_type plan_wave::boundary_g() {
    auto g = [this](const coordinate_t& x) -> Scalar {
        double x1 = x(0), y1 = x(1);
        Scalar res = 1i * k * std::exp(1i * k * (d1 * x(0) + d2 * x(1)));
        if(y1 == 0 && x1 <= 1 && x1 >= 0) {
            res *= (-d1 - 1);
        } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
            res *= (d1 - 1);
        } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
            res *= (d2 - 1);
        } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
            res *= (-d1 - 1);
        }
        return res;
    };
    return g;
}

/*
 * Bessel functions and Hankel functions
 */
Scalar cyl_bessel_j_dx(double v, double x) {
    return x == 0 ? 0 : v * std::cyl_bessel_j(v,x) / x - std::cyl_bessel_j(v+1, x);
}

Scalar cyl_neumann_dx(double v, double x) {
    return v * std::cyl_neumann(v, x) / x - std::cyl_neumann(v+1, x);
}

Scalar hankel_1(double v, double x) {
    return std::cyl_bessel_j(v, x) + 1i * std::cyl_neumann(v, x);
}

Scalar hankel_1_dx(double v, double x) {
    return cyl_bessel_j_dx(v, x) + 1i * cyl_neumann_dx(v, x);
}

/*
 * Fundamental solution
 */
//Scalar fundamental_sol::operator()(const coordinate_t& x) {
//    return hankel_1(k * (x - c).norm());
//}

function_type fundamental_sol::get_fun() {
    return [this](const coordinate_t& x) -> Scalar {
        return hankel_1(0, k * (x - c).norm());
    };
}

function_type fundamental_sol::boundary_g() {
    // u(x) = H0(k*||x-c||)
    // grad u = H' * k * (x-c)/ ||x-c||
    auto g = [this](const coordinate_t& x) -> Scalar {
        double x1 = x(0) - c(0), y1 = x(1) - c(1);
        
        double r = (x - c).norm();

        Scalar u = hankel_1(0, k * r);
        Scalar dudx = hankel_1_dx(0, k * r) * x1 / r;
        Scalar dudy = hankel_1_dx(0, k * r) * y1 / r;
        Scalar res = -1i * k * u;
        if(y1 == 0 && x1 <= 1 && x1 >= 0) {
            // n =  (0, -1)
            res += -1. * dudy;
        } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
            // n = (1, 0)
            res += dudx;
        } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
            // n = (0, 1)
            res += dudy;
        } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
            // n = (-1, 0)
            res += -1. * dudx;
        }
        return res;
    };
    return g;
}

/*
 * Spherical waves
 */

//Scalar Spherical_wave::operator()(const coordinate_t& x){
//    double r = std::sqrt(x(0)^2 + x(1)^2);
//    double phi = std::atan2(x(1), x(0));
//    return std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * l * phi);
//}

function_type Spherical_wave::get_fun() {
    auto u = [this](const coordinate_t& x) -> Scalar {
        double r = std::sqrt(x(0) * x(0) + x(1) * x(1));
        double phi = std::atan2(x(1), x(0));
        return std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * phi * l);
    };
    return u;
}

function_type Spherical_wave::boundary_g() {
    auto g = [this](const coordinate_t& x) -> Scalar {
        double x1 = x(0), y1 = x(1);
        if(x1 == 0 && y1 == 0) {
            return 0;
        }
        double r = std::sqrt(x1 * x1 + y1 * y1);
        double sin_ = y1 / r, cos_ = x1 / r;
        double phi = std::atan2(x(1), x(0));
        
        Scalar u = std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * phi * l);
        Scalar dudx = (cos_ * k * cyl_bessel_j_dx(std::abs(l), k*r) -
                       1i * sin_ * l * std::cyl_bessel_j(std::abs(l), k*r)) * std::exp(1i*phi*l);
        Scalar dudy = (sin_ * k * cyl_bessel_j_dx(std::abs(l), k*r) +
                       1i * cos_ * l * std::cyl_bessel_j(std::abs(l), k*r)) * std::exp(1i*phi*l);
        Scalar res = -1i * k * u;
        if(y1 == 0 && x1 <= 1 && x1 >= 0) {
            // n =  (0, -1)
            res += -1. * dudy;
        } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
            // n = (1, 0)
            res += dudx;
        } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
            // n = (0, 1)
            res += dudy;
        } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
            // n = (-1, 0)
            res += -1. * dudx;
        }
        return res;
    };
    return g;
}
