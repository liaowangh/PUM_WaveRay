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
 * For coarest_mesh.msh, h = 2^{-L-1}, and we can choose k = 2^{L-1}*pi
 */
void test_pumwaveray(int L, int num_wavelayer, std::string& mesh_path, 
    std::string& output_folder, std::vector<std::shared_ptr<HE_sol>>& solutions,
    std::vector<std::string>& sol_name) {
    
    double pi = std::acos(-1.);
    for(int i = 0; i < solutions.size(); ++i) {
        std::vector<double> mesh_width;
        std::vector<double> L2err, H1serr, H1err;
        std::vector<double> wave_nr;

        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();

        // std::cout << sol_name[i] << std::endl;

        for(int l = num_wavelayer; l <= L; ++l) {
            std::vector<int> num_planwaves(l+1);
            num_planwaves[l] = 2;
            for(int j = l - 1; j >= 0; --j) {
                num_planwaves[j] = 2 * num_planwaves[j+1];
            }
            double k_coeff = std::pow(2, l-1);
            ExtendPUM_WaveRay extend_waveray(l, k_coeff*pi, mesh_path, g, u, false, num_planwaves);
            Vec_t fem_sol = extend_waveray.solve_multigrid(l, num_wavelayer, 10, 10); 

            double l2_err = extend_waveray.L2_Err(l, fem_sol, u);
            double h1_serr = extend_waveray.H1_semiErr(l, fem_sol, grad_u);
            double h1_err = std::sqrt(l2_err*l2_err + h1_serr*h1_serr);

            // std::cout << extend_waveray.mesh_width()[l] << " " 
            //           << l2_err << " " << h1_err << std::endl;

            mesh_width.push_back(extend_waveray.mesh_width()[l]);
            wave_nr.push_back(k_coeff);
            L2err.push_back(l2_err);
            H1serr.push_back(h1_serr);
            H1err.push_back(h1_err);
        }
        int num_grids = 1 + num_wavelayer;
        std::vector<std::vector<double>> err_data{mesh_width, wave_nr, L2err, H1err, H1serr};
        std::vector<std::string> data_label{"mesh_width", "  k", "L2_err", "H1_err", "H1_serr"};
        std::string sol_name_mg = std::to_string(num_grids) + "grids_" + sol_name[i];
        print_save_error(err_data, data_label, sol_name_mg, output_folder);
    }
}

int main(){
    std::string square_output = "../result_square/ExtendPUM_WaveRay/";
    std::string square_hole_output = "../result_squarehole/ExtendPUM_WaveRaay/";
    std::string square = "../meshes/square.msh";
    std::string square_hole = "../meshes/square_hole.msh";
    size_type L = 3; // refinement steps
    double k = 2*std::acos(-1.); // wave number
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;

    std::vector<int> num_planwaves(L+1);
    num_planwaves[L] = 2;
    for(int i = L - 1; i >= 0; --i) {
        num_planwaves[i] = 2 * num_planwaves[i+1];
    }
    
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);
 
    std::vector<std::string> sol_name{"waveray_plan_wave", "waveray_fundamental_sol", "waveray_spherical_wave"};
    // test_pumwaveray(4, 1, mesh_path, output_folder, solutions, sol_name);
    
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
        extend_waveray.power_multigird(L, num_wavelayer, 10, 10);
        Vec_t fem_sol = extend_waveray.solve_multigrid(L, num_wavelayer, 10, 10);
        std::cout << extend_waveray.mesh_width()[L] << " "
                  << extend_waveray.L2_Err(L, fem_sol, u) << " " 
                  << extend_waveray.H1_Err(L, fem_sol, u, grad_u) << std::endl;
    }
}