#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/mg_element.h"
#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../Krylov/GMRES.h"
#include "../Pum_WaveRay/HE_FEM.h"
#include "../Pum_WaveRay/PUM_WaveRay.h"
#include "../ExtendPum_WaveRay/ExtendPUM_WaveRay.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

void wave_ray(HE_LagrangeO1& he_O1, HE_FEM& pum, int wave_start, int wave_coarselayers,
    int ray_start, int ray_coarselayers, double k, FHandle_t u) {

    auto eq_pair = he_O1.build_equation(wave_start);
    SpMat_t wave_A(eq_pair.first.makeSparse());
  
    mg_element wave_mg(he_O1, wave_start, wave_coarselayers);
    mg_element ray_mg(pum, ray_start, ray_coarselayers);

    /*********************************************************************/
    std::vector<double> L2_vk, L2_ek;
    Vec_t v = Vec_t::Random(wave_A.rows());
    Vec_t uh = he_O1.solve(wave_start);  // finite element solution
    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };
    
    std::cout << std::setw(11) << "||v-uh||_2" << std::setw(11) << "||v-u||_2" << std::endl;
    for(int j = 0; j < 10; ++j) {
        std::cout << std::setw(11) << he_O1.L2_Err(wave_start, v - uh, zero_fun) << " ";
        std::cout << std::setw(11) << he_O1.L2_Err(wave_start, v, u) << std::endl;
        L2_vk.push_back(he_O1.L2_Err(wave_start, v, zero_fun));
        L2_ek.push_back(he_O1.L2_Err(wave_start, v - uh, zero_fun));
        wave_cycle(v, eq_pair.second, wave_mg, ray_mg, k, wave_start, ray_start);
    }

    std::cout << "||u-uh||_2 = " << he_O1.L2_Err(wave_start, uh, u) << std::endl;
    std::cout << "||v_{k+1}||/||v_k||" << std::endl;
    std::cout << "j " 
        << std::setw(20) << "||v_{j+1}||/||v_j||" 
        << std::setw(20) << "||e_{j+1}||/||e_j||" << std::endl;
    for(int j = 0; j + 1 < L2_vk.size(); ++j) {
        std::cout << j << " " << std::setw(20) << L2_vk[j+1] / L2_vk[j] 
                       << " " << std::setw(20) << L2_ek[j+1] / L2_ek[j] << std::endl;
    }
}

double power_wave_ray(HE_LagrangeO1& he_O1, HE_FEM& pum, int wave_start, int wave_coarselayers,
    int ray_start, int ray_coarselayers, double k, bool verbose = true) {

    auto eq_pair = he_O1.build_equation(wave_start);
    SpMat_t wave_A(eq_pair.first.makeSparse());
  
    mg_element wave_mg(he_O1, wave_start, wave_coarselayers);
    mg_element ray_mg(pum, ray_start, ray_coarselayers);

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
        wave_cycle(v, zero_vec, wave_mg, ray_mg, k, wave_start, ray_start);    
    
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
    int wave_L = 5, ray_L = 4;
    int wave_coarselayers = 5, ray_coarselayers = 1;

    std::string square = "../meshes/square.msh";
    std::string square_hole2 = "../meshes/square_hole2.msh";
    std::string triangle_hole = "../meshes/triangle_hole.msh";

    std::string output_folder = "../result/waveray_factor/";

    std::vector<std::pair<std::string, bool>> 
        mesh{{square, false}, {square_hole2, true}, {triangle_hole, true}};
    std::vector<int> num_planwaves(ray_L+1);
    num_planwaves[ray_L] = 2;
    for(int i = ray_L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;

    std::vector<std::vector<double>> pum_factors(4, std::vector<double>(wave_number.size(), 0));
    std::vector<std::vector<double>> epum_factors(4, std::vector<double>(wave_number.size(), 0));
    for(int i = 0; i < wave_number.size(); ++i) {
        double k = wave_number[i];
        pum_factors[3][i] = k;
        epum_factors[3][i] = k;

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
            
            // PUM_WaveRay pum(ray_L, k, m.first, g, u, m.second, num_planwaves, 30);
            // double rho1 = power_wave_ray(he_O1, pum, wave_L, wave_coarselayers, 
            //     ray_L, ray_coarselayers, k, false);
            ExtendPUM_WaveRay epum(ray_L, k, m.first, g, u, m.second, num_planwaves, 30);
            double rho2 = power_wave_ray(he_O1, epum, wave_L, wave_coarselayers, 
                ray_L, ray_coarselayers, k, false);
            // std::cout << k << " " << rho1 << " " << rho2 << std::endl;
            // std::cout << k << " " << rho2 << std::endl;
            // pum_factors[j][i] = rho1;
            epum_factors[j][i] = rho2;
        }  
    }
    std::string suffix = std::to_string(wave_L)+std::to_string(wave_coarselayers) 
        + std::to_string(ray_L) + std::to_string(ray_coarselayers);
    std::vector<std::string> labels{"square", "square_hole", "triangle_hole", "k"};
    // print_save_error(pum_factors, labels, "pum_planwave_" + suffix, output_folder);
    print_save_error(epum_factors, labels, "epum_planwave_" + suffix, output_folder);
}

int main(){
    // std::string square_output = "../result_square/ExtendPUM_WaveRay/";
    // std::string square_hole_output = "../result_squarehole/ExtendPUM_WaveRaay/";
    // std::string square = "../meshes/square.msh";
    // std::string square_hole = "../meshes/square_hole.msh";
    // std::string square_hole2 = "../meshes/square_hole2.msh";
    // std::string triangle_hole = "../meshes/triangle_hole.msh";

    // std::vector<std::pair<std::string, bool>> 
    //     mesh{{square, false}, {square_hole2, true}, {triangle_hole, true}};

    // size_type wave_L = 6, ray_L = 5; // refinement steps
    // double k = 23; // wave number
    // std::vector<int> num_planwaves(ray_L+1);
    // num_planwaves[ray_L] = 2;
    // for(int i = ray_L - 1; i >= 0; --i) {
    //     num_planwaves[i] = 2 * num_planwaves[i+1];
    // }

    // plan_wave sol(k, 0.8, 0.6);
    // auto u = sol.get_fun();
    // auto grad_u = sol.get_gradient();
    // auto g = sol.boundary_g();
    // int wave_coarselayers = 6, ray_coarselayers = 1;
    // for(int i = 0; i < mesh.size(); ++i) {
    //     if(i != 0) continue;
    //     auto m = mesh[i];
    //     HE_LagrangeO1 he_O1(wave_L, k, m.first, g, u, m.second, 30);
    //     PUM_WaveRay pum(ray_L, k, m.first, g, u, m.second, num_planwaves, 30);
    //     ExtendPUM_WaveRay epum(ray_L, k, m.first, g, u, m.second, num_planwaves, 30);

    //     // wave_ray(he_O1, pum, wave_L, wave_coarselayers, ray_L, ray_coarselayers, k, u);
    //     // power_wave_ray(he_O1, pum, wave_L, wave_coarselayers, ray_L, ray_coarselayers, k);

    //     wave_ray(he_O1, epum, wave_L, wave_coarselayers, ray_L, ray_coarselayers, k, u);
    //     power_wave_ray(he_O1, epum, wave_L, wave_coarselayers, ray_L, ray_coarselayers, k);
    // }  
    wave_ray_factor();
}