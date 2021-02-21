#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../../utils/HE_solution.h"
#include "../../utils/utils.h"
#include "../ExtendPUM_WaveRay.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

/*
 * For square_hole.msh, h = 2^{-L-1}, and we can choose k = 2^{L-1}*pi
 * For square.msh, h_l = =2^{-l}, and we choose k = 2^{L-1}, then h_L * k = 0.5
 */
void test_Extend_WaveRay(int num_wavelayer, const std::string& mesh_path, bool hole_exist, size_type L, 
    std::vector<double> wave_number, const std::string& output_folder, 
    std::shared_ptr<HE_sol> sol, const std::string& sol_name);

int main(){
    std::string square_output = "../result_square/ExtendPUM_WaveRay/";
    std::string square_hole_output = "../result_squarehole/ExtendPUM_WaveRaay/";
    std::string square = "../meshes/square.msh";
    std::string square_hole = "../meshes/square_hole.msh";
    size_type L = 5; // refinement steps
    double k = 4; // wave number

    std::vector<double> wave_nrs{2, 4, 6, 8};

    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);
 
    std::vector<std::string> sol_name{"waveray_plan_wave", "waveray_fundamental_sol", "waveray_spherical_wave"};
    // test_Extend_WaveRay(1, square, false, L, wave_nrs, square_output, solutions[0], sol_name[0]);

    std::vector<int> num_planwaves(L+1);
    num_planwaves[L] = 2;
    for(int i = L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }
    for(int i = 0; i < solutions.size(); ++i) {
        if(i > 0){
            continue;
        }
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();
        ExtendPUM_WaveRay extend_waveray(L, k, square, g, u, false, num_planwaves);
        // std::string prefix = "k" + std::to_string(int(k)) + "_";
        // test_solve(extend_waveray, prefix+sol_name[i], output_folder, L, u, grad_u);
        // test_prolongation(extend_waveray, L);
    
        // auto eq_pair = extend_waveray.build_equation(L);
        // SpMat_t A = eq_pair.first.makeSparse();
        // auto eigen_pair = power_GS(A, 1);
        // extend_waveray.vector_vtk(L, eigen_pair.first, "GS_mode");
    
        int num_wavelayer = 1;
        // extend_waveray.power_multigird(L, num_wavelayer, 3, 3);
        Vec_t fem_sol = extend_waveray.solve_multigrid(L, num_wavelayer, 3, 3);
        std::cout << extend_waveray.mesh_width()[L] << " "
                  << extend_waveray.L2_Err(L, fem_sol, u) << " " 
                  << extend_waveray.H1_Err(L, fem_sol, u, grad_u) << std::endl;
    }
}

void test_Extend_WaveRay(int num_wavelayer, const std::string& mesh_path, bool hole_exist, size_type L, 
    std::vector<double> wave_number, const std::string& output_folder, 
    std::shared_ptr<HE_sol> sol, const std::string& sol_name) {
    
    auto u = sol->get_fun();
    auto grad_u = sol->get_gradient();
    auto g = sol->boundary_g();

    std::vector<std::vector<double>> L2err(wave_number.size()), H1serr(wave_number.size());
    std::vector<std::string> data_label;
    for(int i = 0; i < wave_number.size(); ++i) {
        int wave_nr = wave_number[i];
        data_label.push_back(std::to_string(int(wave_nr)));

        for(int l = num_wavelayer; l <= L; ++l) {

            std::vector<int> num_planwaves(l+1);
            num_planwaves[l] = 2;
            for(int j = l - 1; j >= 0; --j) {
                num_planwaves[j] = 2 * num_planwaves[j+1];
            }

            ExtendPUM_WaveRay extend_waveray(l, wave_nr, mesh_path, g, u, hole_exist, num_planwaves, 20);

            auto fe_sol = extend_waveray.solve_multigrid(l, num_wavelayer, 3, 3);
            double l2_err = extend_waveray.L2_Err(l, fe_sol, u);
            double h1_serr = extend_waveray.H1_semiErr(l, fe_sol, grad_u);
            
            // ndofs.push_back(fe_sol.size());
            L2err[i].push_back(l2_err);
            H1serr[i].push_back(h1_serr);
        }
    }
    tabular_output(L2err, data_label, sol_name + "_L2err", output_folder, false);
    tabular_output(H1serr, data_label, sol_name + "_H1serr", output_folder, false);
}