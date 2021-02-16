#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../LagrangeO1/HE_LagrangeO1.h"
#include "../Pum_WaveRay/PUM_WaveRay.h"
#include "../ExtendPum_WaveRay/ExtendPUM_WaveRay.h"

using Scalar = std::complex<double>;
using size_type = unsigned int;
using coordinate_t = Eigen::Vector2d;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using triplet_t = Eigen::Triplet<Scalar>;
using SpMat_t = Eigen::SparseMatrix<Scalar>;
using FHandle_t = std::function<Scalar(const coordinate_t&)>;
using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;

using namespace std::complex_literals;

void test_prolongation(HE_FEM& he_fem, const Vec_t& vec_f, int L, const FHandle_t& f);

int main() {
    // std::string mesh_path = "../meshes/square_hole.msh";
    std::string mesh_path = "../meshes/square.msh";
    size_type L = 3; // refinement steps
    double k = std::acos(-1.); // wave number

    std::vector<int> num_planwaves(L+1);
    num_planwaves[L] = 2;
    for(int i = L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }
    int N0 = num_planwaves[0];

    plan_wave pw(k, 0.8, 0.6);
    auto u = pw.get_fun();
    auto g = pw.boundary_g();

    HE_LagrangeO1 he_O1(L, k, mesh_path, g, u, false);
    ExtendPUM_WaveRay extend_waveray(L, k, mesh_path, g, u, false, num_planwaves, 20);
    PUM_WaveRay waveray(L, k, mesh_path, g, u, false, num_planwaves, 20);

    /************************************************************************/
    // std::cout << "Test prolongation operator between Lagrangian finite element spaces." << std::endl;
    // // auto test_f = [](const coordinate_t& x)->Scalar {
    // //     return 2.0*x(0) + 3.0*x(1);
    // // };
    // Vec_t vec_f = he_O1.fun_in_vec(0, u);
    // test_prolongation(he_O1, vec_f, L, u);

    /************************************************************************/
    // std::cout << "Test prolongation operator using best approximation between PUM sapces." << std::endl;
    // auto vec_u = waveray.fun_in_vec(0, u);
    // // Vec_t vec_u = Vec_t::Zero(5*N0);
    // // for(int i = 0; i < 5*N0; i += N0) {
    // //     vec_u(i) = 1.0;
    // // }
    // test_prolongation(waveray, vec_u, L, u);

    /************************************************************************/
    std::cout << "Test prolongation operator using best approximation between extend PUM sapces." << std::endl;
    auto vec_u2 = extend_waveray.fun_in_vec(0, u);
    // Vec_t vec_u2 = Vec_t::Zero(5*(N0+1));
    // for(int i = 1; i < 5*(N0+1); i += (N0+1)) {
    //     vec_u2(i) = 1.0;
    // }
    test_prolongation(extend_waveray, vec_u2, L, u);
}

void test_prolongation(HE_FEM& he_fem, const Vec_t& vec_f, int L, const FHandle_t& f) {
    std::vector<SpMat_t> pro_op(L);
    for(int i = 0; i < L; ++i) {
        pro_op[i] = he_fem.prolongation(i);
        std::cout << i << " :[" << pro_op[i].rows() << "," 
            << pro_op[i].cols() << "]" << std::endl;
        // std::cout << pro_op[i] << std::endl;
    }

    std::vector<Vec_t> vec_in_mesh(L+1);
    vec_in_mesh[0] = vec_f;
    for(int i = 1; i <= L; ++i) {
        vec_in_mesh[i] = pro_op[i-1] * vec_in_mesh[i-1];
    }

    // for(int i = 0; i <= L; ++i){
    //     std::cout << vec_in_mesh[i].size() << std::endl;
    //     std::cout << vec_in_mesh[i] << std::endl;
    // }
    // std::cout << vec_in_mesh[1];
    // std::cout << pro_op[0] << std::endl;

    std::cout << std::left << std::setw(7) << "level" 
        << std::setw(15) << "||v-f||" 
        << std::setw(15) << "||v-f||/||f||"
        << std::setw(15) << "||v-uh||" << std::endl;
    std::cout << std::scientific << std::setprecision(2);
    for(int i = 0; i <= L; ++i) {
        auto true_vec = he_fem.fun_in_vec(i, f);
        double l2err = he_fem.L2_Err(i, vec_in_mesh[i], f);
        std::cout << std::left << std::setw(7) << i 
            << std::setw(15) << l2err 
            << std::setw(15) << l2err / (L2_norm(he_fem.getmesh(i), f, 20))
            << std::setw(15) << (true_vec - vec_in_mesh[i]).norm()
            << std::endl;
    }
}