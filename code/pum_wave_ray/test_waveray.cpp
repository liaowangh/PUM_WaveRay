#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "PUM_WaveRay.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

void test_pumwaveray(int k, int L, int num_wavelayer, std::string& mesh_path, 
    std::string& output_folder, std::vector<std::shared_ptr<HE_sol>>& solutions,
    std::vector<std::string>& sol_name) {
    for(int i = 0; i < solutions.size(); ++i) {
        std::vector<double> mesh_width;
        std::vector<double> L2err, H1serr, H1err;

        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();
        for(int l = num_wavelayer; l <= L; ++l) {
            std::vector<int> num_planwaves(l+1);
            num_planwaves[l] = 2;
            for(int i = l - 1; i >= 0; --i) {
                num_planwaves[i] = 2 * num_planwaves[i+1];
            }
            PUM_WaveRay he_waveray(l, k, mesh_path, g, u, true, num_planwaves);
            Vec_t fe_sol = he_waveray.solve_multigrid(l, num_wavelayer, 10, 10); 

            double l2_err = he_waveray.L2_Err(l, fe_sol, u);
            double h1_serr = he_waveray.H1_semiErr(l, fe_sol, grad_u);
            double h1_err = std::sqrt(l2_err*l2_err + h1_serr*h1_serr);

            // ndofs.push_back(fe_sol.size());
            mesh_width.push_back(he_waveray.mesh_width()[l]);
            L2err.push_back(l2_err);
            H1serr.push_back(h1_serr);
            H1err.push_back(h1_err);
        }
        int num_grids = 1 + num_wavelayer;
        std::vector<std::vector<double>> err_data{mesh_width, L2err, H1err, H1serr};
        std::vector<std::string> data_label{"mesh_width", "L2_err", "H1_err", "H1_serr"};
        std::string sol_name_mg = std::to_string(num_grids) + "grids_" + sol_name[i];
        print_save_error(err_data, data_label, sol_name_mg, output_folder);
    }
}

int main(){
    // mesh path
    boost::filesystem::path here = __FILE__;
    std::string output_folder = "../plot_err/pum_waveray/";
    auto mesh_path = (here.parent_path().parent_path() / ("meshes/coarest_mesh.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/tri2.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/square.msh")).string(); 
    size_type L = 4; // refinement steps
    double k = 2; // wave number
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
    test_pumwaveray(k, L, 1, mesh_path, output_folder, solutions, sol_name);
    /*
    for(int i = 0; i < solutions.size(); ++i) {
        if(i > 0){
            continue;
        }
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();
        PUM_WaveRay he_waveray(L, k, mesh_path, g, u, true, num_planwaves);
        std::string prefix = "k" + std::to_string(int(k)) + "_";
        // test_solve(he_waveray, prefix+sol_name[i], output_folder, L, u, grad_u);
        // test_prolongation(he_waveray, L);
        // he_waveray.power_multigird(2, 1, 10, 10);

        int num_wavelayer = 1;
        std::cout << "Start v cycle;" << std::endl;
        Vec_t fem_sol = he_waveray.solve_multigrid(L, num_wavelayer, 10, 10);
        std::cout << he_waveray.L2_Err(L, fem_sol, u) << std::endl;

        // std::cout << std::endl << fem_sol << std::endl;

        // std::cout << std::endl;
    }
    */
}