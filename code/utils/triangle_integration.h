#include <complex>
#include <cmath>
#include <vector>

#include <Eigen/Core>

using data_type = std::complex<double>;

// \int_a^b exp(tx)dx
data_type int_exp(double a, double b);

// \int_a^b x*exp(tx)dx
data_type int_xexp(double a, double b, data_type t);

// \int_a^b x*x*exp(tx)dx
data_type int_x2exp(double a, double b, data_type t);

// Integration over unit triangle K = covex{[0,0], [1,0], [0,1]}

// \int_K exp(ik(d1*x+d2*y))dxdy
data_type intK_exp(double k, double d1, double d2);

// \int_K x*exp(ik(d1*x+d2*y))dxdy
data_type intK_x_exp(double k, double d1, double d2);

// \int_K yexp(ik(d1*x+d2*y))dxdy
data_type intK_y_exp(double k, double d1, double d2);

// \int_K x^2exp(ik(d1*x + d2*y))dxdy
data_type intK_x2_exp(double k, double d1, double d2);

// \int_K y^2exp(ik(d1*x+d2*y))dxdy
data_type intK_y2_exp(double k, double d1, double d2);

// \int_K x*y*exp(ik(d1*x+d2*y))dxdy
data_type intK_xy_exp(double k, double d1, double d2);

// \int_K (a0x^2+a1y^2+a2xy+a3x+a4y+a5)exp(ik(d1*x+d2*y))dxdy
data_type int_fxy_exp(double k, std::vector<data_type> a, Eigen::Vector2d d);


