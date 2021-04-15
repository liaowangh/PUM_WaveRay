#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../../utils/mg_element.h"
#include "../../utils/HE_solution.h"
#include "../../LagrangeO1/HE_LagrangeO1.h"
#include "../../utils/utils.h"
#include "../../Pum_WaveRay/HE_FEM.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

void O1_wavecycle(Vec_t& v, Vec_t& phi, const mg_element& wave_mg,
    double k, int wave_L) {

    const std::vector<SpMat_t>& wave_I = wave_mg.I;
    std::vector<SpMat_t> wave_op = wave_mg.op;
    const std::vector<int>& block_size = wave_mg.block_size;
    const std::vector<double> wave_ms = wave_mg.mw;

    int nu1 = 2, nu2 = 2;
    int wave_coarselayers = wave_I.size();
    std::vector<int> op_size(wave_coarselayers + 1);
    for(int i = 0; i <= wave_coarselayers; ++i) {
        op_size[i] = wave_op[i].rows();
    }
    std::vector<Vec_t> initial(wave_coarselayers + 1), rhs_vec(wave_coarselayers + 1);

    /************ first leg of wave cycle *************/
    initial[wave_coarselayers] = v;
    rhs_vec[wave_coarselayers] = phi;
    // initial guess on coarser mesh are all zero
    for(int i = 0; i < wave_coarselayers; ++i) {
        initial[i] = Vec_t::Zero(op_size[i]);
    }
    for(int i = wave_coarselayers; i > 0; --i) {
        Gaussian_Seidel(wave_op[i], rhs_vec[i], initial[i], 1, nu1);
        rhs_vec[i-1] = wave_I[i-1].transpose() * (rhs_vec[i] - wave_op[i] * initial[i]);
    }

    Eigen::SparseLU<SpMat_t> solver;
    solver.compute(wave_op[0]);
    initial[0] = solver.solve(rhs_vec[0]);

    /************ second leg of wave cycle *************/
    for(int i = 1; i <= wave_coarselayers; ++i) {
        initial[i] += wave_I[i-1] * initial[i-1];
        Gaussian_Seidel(wave_op[i], rhs_vec[i], initial[i], 1, nu2);
    }
    v = initial[wave_coarselayers];
}

double power_O1(HE_LagrangeO1& he_O1, int wave_start, int wave_coarselayers,
    double k, bool verbose = true) {

    auto eq_pair = he_O1.build_equation(wave_start);
    SpMat_t wave_A(eq_pair.first.makeSparse());
  
    mg_element wave_mg(he_O1, wave_start, wave_coarselayers);

    /********* start power iteration **********/
    int N = wave_A.rows();
    Vec_t v = Vec_t::Random(N);
    v.normalize();
    Vec_t old_v;
    Vec_t zero_vec = Vec_t::Zero(N);
    Scalar lambda = 0;
    Scalar old_lambda;
    int cnt = 0;
    
    if(verbose) {
        std::cout << std::left << std::setw(10) << "Iteration" 
            << std::setw(20) << "residual_norm" << std::endl;
    }
    
    while(true) {
        cnt++;
        old_v = v;
        old_lambda = lambda;
        O1_wavecycle(v, zero_vec, wave_mg, k, wave_start);
    
        lambda = old_v.dot(v);  // domainant eigenvalue
        
        auto r = v - lambda * old_v;
        double r_norm = r.norm();
        v.normalize();
    
        if(verbose && cnt % 10 == 0) {
            std::cout << std::left << std::setw(10) << cnt 
                << std::setw(20) << r_norm  
                << std::setw(20) << (v - old_v).norm()
                << std::setw(5)  << std::abs(old_lambda - lambda)
                << std::endl;
        }
        if(cnt >= 3 && std::abs(old_lambda - lambda) < 0.001) {
            break;
        }
        if(cnt >= 200) {
            if(verbose) std::cout << "Power iteration for multigrid doesn't converge." << std::endl;
            break;
        }
    }
    if(verbose) {
        std::cout << "Number of iterations: " << cnt << std::endl;
        std::cout << "Domainant eigenvalue by power iteration: " << std::abs(lambda) << std::endl;
    }
    return std::abs(lambda); 
}

void wave_ray_factor() {
    std::vector<double> wave_number;
    for(int i = 1; i <= 10; ++i) {
        wave_number.push_back(i);
    }
    int wave_L = 5;
    int wave_coarselayers = 3;

    std::string square = "../meshes/square.msh";
    std::string square_hole2 = "../meshes/square_hole2.msh";
    std::string triangle_hole = "../meshes/triangle_hole.msh";

    std::string output_folder = "../result/O1_factor/";

    std::vector<std::pair<std::string, bool>> 
        mesh{{square, false}, {square_hole2, true}, {triangle_hole, true}};
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;

    std::vector<std::vector<double>> factors(mesh.size() + 1, 
        std::vector<double>(wave_number.size(), 0));
    for(int i = 0; i < wave_number.size(); ++i) {
        double k = wave_number[i];
        factors[mesh.size()][i] = k;

        std::vector<std::shared_ptr<HE_sol>> solutions(3);
        solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
        solutions[1] = std::make_shared<fundamental_sol>(k, c);
        solutions[2] = std::make_shared<Spherical_wave>(k, 2); 

        for(int j = 0; j < mesh.size(); ++j) {

            auto u = solutions[0]->get_fun();
            auto grad_u = solutions[0]->get_gradient();
            auto g = solutions[0]->boundary_g();
            
            auto m = mesh[j];
            HE_LagrangeO1 he_O1(wave_L, k, m.first, g, u, m.second, 30);
            
            double rho = power_O1(he_O1, wave_L, wave_coarselayers, k, false);
            factors[j][i] = rho;
        }  
    }
    std::string suffix = std::to_string(wave_L)+std::to_string(wave_coarselayers);
    std::vector<std::string> labels{"square", "square_hole", "triangle_hole", "k"};
    print_save_error(factors, labels, "O1" + suffix, output_folder);
}

int main(){
    wave_ray_factor();
}

// int main(){
//     std::string square_output = "../result_square/ExtendPUM_WaveRay/";
//     std::string square_hole_output = "../result_squarehole/ExtendPUM_WaveRaay/";
//     std::string square = "../meshes/square.msh";
//     std::string square_hole = "../meshes/square_hole.msh";
//     std::string square_hole2 = "../meshes/square_hole2.msh";
//     std::string triangle_hole = "../meshes/triangle_hole.msh";

//     std::vector<std::pair<std::string, bool>> 
//         mesh{{square, false}, {square_hole2, true}, {triangle_hole, true}};

//     size_type wave_L = 6; // refinement steps
//     double k = 23; // wave number

//     plan_wave sol(k, 0.8, 0.6);
//     auto u = sol.get_fun();
//     auto grad_u = sol.get_gradient();
//     auto g = sol.boundary_g();
//     int wave_coarselayers = 6;
//     for(int i = 0; i < mesh.size(); ++i) {
//         if(i != 0) continue;
//         auto m = mesh[i];
//         HE_LagrangeO1 he_O1(wave_L, k, m.first, g, u, m.second, 30);
//         power_O1(he_O1, wave_L, wave_coarselayers, k, true);
//     }  
// }