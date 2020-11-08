#include <iostream>
#include <complex>
#include <cmath>
#include <vector>

#include "triangle_integration.h"

using namespace std::complex_literals;
using data_type = std::complex<double>;

/*
 * \int_a^b exp(tx)dx
 */
data_type int_exp(double a, double b, Triangle_Integratioin::data_type t) {
    return t == 0 ? b - a : (std::exp(t*b) - std::exp(t*a)) / t;
}

/*
 * \int_a^b x*exp(tx)dx
 */
data_type int_xexp(double a, double b, Triangle_Integratioin::data_type t) {
    return t == 0 ? (b*b - a*a) / 2 :
        (b*std::exp(t*b) - a*std::exp(t*a)) / t - (std::exp(t*b) - std::exp(t*a)) / (t*t);
}

/*
 * \int_a^b x*x*exp(tx)dx
 */
data_type int_x2exp(double a, double b, Triangle_Integratioin::data_type t) {
    return t == 0 ? (b*b*b - a*a*a) / 3 :
        (b*b*std::exp(t*b) - a*a*std::exp(t*a)) / t - 2 * (b*std::exp(t*b) - a*std::exp(t*a)) / (t*t) +
        2*(std::exp(t*b) - std::exp(t*a)) / std::pow(t, 3);
}


/*
 * \int_K exp(ik(d1*x+d2*y))dxdy
 */
data_type intK_exp(double k, double d1, double d2) {
    double dd = d2 - d1;
    Triangle_Integratioin::data_type res = 0;
    if(d1 == d2) {
        res = (-1i / (k*d1) + 1 / (k*k*d1*d1)) * std::exp(1i*k*d1) - 1/(k*k*d1*d1);
    } else {
        res = (-std::exp(1i*k*d2) + std::exp(1i*k*d1)) / (k*k*d1*dd) +
        (std::exp(1i*k*d2) - 1) / (k*k*d1*d2);
    }
    return res;
}

/*
 * \int_K xexp(ik(d1*x+d2*y))dxdy
 */
data_type intK_x_exp(double k, double d1, double d2) {
    Triangle_Integratioin::data_type res = 0;
    double dd = d2 - d1;
    if(d1 == d2) {
        res = (-1 / (2*k*d1) + 1 / (k*k*d1*d1) + 1i / std::pow(k*d1, 3)) *
            std::exp(1i*k*d1) - 1i / std::exp(1i*k*d1);
    } else {
        res = std::exp(1i*k*d1) / (k*k*d1*dd) +
              1i*std::exp(1i*k*d2) / (k*k*k*d1*d1*d2) +
              (1i/(k*k*k*dd*dd*d1)-1/(k*k*d1*dd)-1i/(k*k*k*d1*d1*dd)) * (std::exp(1i*k*d2) - std::exp(1i*k*d1)) -
              1i/(k*k*k*d1*d1*d2);
    }
    return res;
}

/*
 * \int_K yexp(ik(d1*x+d2*y))dxdy
 */
data_type intK_y_exp(double k, double d1, double d2) {
    return intK_x_exp(d2, d1);
}


/*
 * \int_K x^2exp(ik(d1*x + d2*y))dxdy
 */
data_type intK_x2_exp(double k, double d1, double d2) {
    Triangle_Integratioin::data_type res = 0;
    if(d1 == d2) {
        res += 1 / (3i * k * d1) + 1 / std::pow(k * d1, 2);
        res += 2i / std::pow(k * d1, 3) - 2 / std::pow(k * d1, 4);
        res *= std::exp(1i * k * d1);
        res += 2i / std::pow(k * d1, 3);
        return res;
    }
    double dd = d2 - d1;
    res += -std::exp(1i*k*d2) / (k*k*d1*(d2-d1)) -
        (2i/(k*k*k*dd*dd*d1) - 2/(k*k*dd*d1)) * std::exp(1i*k*d2);
    res += (2/(std::pow(k*dd,3)*k*d1) + 2i/(k*k*k*dd*dd*d1) - 1/(k*k*dd*d1)) *
        (std::exp(1i*k*d2) - std::exp(1i*k*d1));
    res += 2i*std::exp(1i*k*d2) / (k*k*k*d1*d1*dd) -
        (4/(std::pow(k*d1*dd,2)*k*k)+2i/(k*k*k*d1*d1*dd)) *
        (std::exp(1i*k*d2) - std::exp(1i*k*d1));
    res += 1 / (std::pow(k*d1,3)*k*dd)*(std::exp(1i*k*d2) - std::exp(1i*k*d1)) -
        2 / (std::pow(k*d1,3)*k*d2) * (std::exp(1i*k*d2) - 1);
    return res;
}

/*
 * \int_K y^2exp(ik(d1*x+d2*y))dxdy
 */
data_type intK_y2_exp(double k, double d1, double d2) {
    return intK_x2_exp(d2, d1);
}

/*
 * \int_K x*y*exp(ik(d1*x+d2*y))dxdy
 */
data_type intK_xy_exp(double k, double d1, double d2) {
    return 1i*std::exp(1i*k*d1) / (k*d1) * int_x2_exp(0, 1, 1i*k*(d2 - d1)) +
        (-1i*std::exp(1i*k*d1) / (k*d1) + std::exp(1i*k*d1) / std::pow(k*d1,2)) * int_x_exp(0, 1, 1i*k*(d2-d1)) -
        int_x_exp(0, 1, 1i*k*d2) / std::pow(k*d1, 2);
}

/*
 * \int_K (a0x^2+a1y^2+a2xy+a3x+a4y+a5)exp(ik(d1*x+d2*y))dxdy
 */
data_type int_fxy_exp(double k, std::vector<data_type> a, Eigen::Vector2d d){
    return
        a[0] * intK_x2_exp(d(0), d(1)) +
        a[1] * intK_y2_exp(d(0), d(1)) +
        a[2] * intK_xy_exp(d(0), d(1)) +
        a[3] * intK_x_exp(d(0), d(1)) +
        a[4] * intK_y_exp(d(0), d(1)) +
        a[5] * intK_exp(d(0), d(1));
}
