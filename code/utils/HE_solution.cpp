#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "wave_solution"

using namespace std::complex_literals;

plan_wave::plan_wave(double k_, double d1_, double d2_): k(k_), d1(d1_), d2(d2_){
    g = [&k](const Eigen::Vector2d& x) -> Scalar {
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
}

Scalar cyl_bessel_j_dx(double v, double x) {
    return x == 0 ? 0 : v / x * std::cyl_bessel_j(v,x) - std::cyl_bessel_j(v+1, x);
}

Spherical_wave::Shperical_wave(double k_, int l_): k(k_), l(l_){
    g = [&k](const Eigen::Vector2d& x) -> Scalar {
        double x1 = x(0), y1 = x(1);
        double r = std::sqrt(x1 * x1 + y1 * y1);
        double sin_ = y1 / r, cos_ = x1 / r;
        double phi = std::atan2(x(1), x(0));
        
        
        Scalar u = std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * l * phi)
        Scalar dudx = (cos_ * k * cyl_bessel_j_dx(std::abs(l), k*r) -
                        1i * l * sin_ * cyl_bessel_j(std::abs(l), k*r)) * std::exp(i*l*phi);
        Scalar dudy = (sin_ * k * cyl_bessel_j_dx(std::abs(l), k*r) +
                       1i * l * cos_ * cyl_bessel_j(std::abs(l), k*r)) * std::exp(i*l*phi);
        Scalar res = -1i * k * u;
        if(y1 == 0 && x1 <= 1 && x1 >= 0) {
            // n =  (0, -1)
            res += -1 * dudy;
        } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
            // n = (1, 0)
            res += dudx;
        } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
            // n = (0, 1)
            res += dudy;
        } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
            // n = (-1, 0)
            res += -1 * dudx;
        }
        return res;
    };
}

Scalar Spherical_wave::operator()(const Eigen::Vector2d& x){
    double r = std::sqrt(x(0)^2 + x(1)^2);
    double phi = std::atan2(x(1), x(0));
    return std::cyl_bessel_j(std::abs(l), k*r) * std::exp(1i * l * phi);
}
