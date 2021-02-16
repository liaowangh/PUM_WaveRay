#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../HE_ExtendPUM.h"
#include "../../utils/HE_solution.h"
#include "../../utils/utils.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

void resolution_test(const std::string& mesh_path, bool hole_exist, size_type L, double wave_nr, 
    std::vector<int> num_waves, const std::string& output_folder, 
    std::shared_ptr<HE_sol> sol, const std::string& sol_name);

void tabular_output(std::vector<std::vector<double>>& data, 
    std::vector<std::string>& data_label, const std::string& sol_name, 
    const std::string& output_folder);

/*
 * Convergence studies
 *  different wave numbers: k = 6, 20, 60
 *  on sequences of meshes (refinement steps: 5)
 *  different number of plan waves (0, 3, 5, 7, 9, 11, 13)
 *  compute L2 error norm and H1 error semi-norm
 */
int main() {
    std::string square_hole = "../meshes/square_hole.msh";
    std::string square = "../meshes/square.msh";
    std::string square_hole_output = "../result_squarehole/ExtendPUM/";
    std::string square_output = "../result_square/ExtendPUM/";
    int L = 5; // refinement steps
    std::vector<int> number_waves{0, 3, 5, 7, 9, 11, 13};
    // std::vector<int> number_waves{0};
    
    std::vector<double> wave_number{6, 20, 60};

    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;
    
    std::vector<std::string> sol_name{"plan_wave", "fundamental_sol", "spherical_wave"};
    for(auto k: wave_number) {
        std::vector<std::shared_ptr<HE_sol>> solutions(3);
        solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
        solutions[1] = std::make_shared<fundamental_sol>(k, c);
        solutions[2] = std::make_shared<Spherical_wave>(k, 2);
        for(int i = 0; i < solutions.size(); ++i) {
            if(i > 0) { continue; }
            std::string str = "resolution_k" + std::to_string(int(k)) + "_" + sol_name[i];
            resolution_test(square_hole, true, L, k, number_waves, square_hole_output, solutions[i], str);
            // resolution_test(square, false, L, k, number_waves, square_output, solutions[i], str);
            std::cout << std::endl;
        }
    }
}

void resolution_test(const std::string& mesh_path, bool hole_exist, size_type L, double wave_nr, 
    std::vector<int> num_waves, const std::string& output_folder, 
    std::shared_ptr<HE_sol> sol, const std::string& sol_name) {
    
    auto u = sol->get_fun();
    auto grad_u = sol->get_gradient();
    auto g = sol->boundary_g();

    std::vector<std::vector<double>> L2err(num_waves.size()), H1serr(num_waves.size());
    std::vector<std::string> data_label;
    for(int i = 0; i < num_waves.size(); ++i) {
        int nr_waves = num_waves[i];
        data_label.push_back(std::to_string(nr_waves));
        HE_ExtendPUM he_epum(L, wave_nr, mesh_path, g, u, hole_exist, std::vector<int>(L+1, nr_waves), 20);

        for(int l = 0; l <= L; ++l) {
            auto fe_sol = he_epum.solve(l);
            double l2_err = he_epum.L2_Err(l, fe_sol, u);
            double h1_serr = he_epum.H1_semiErr(l, fe_sol, grad_u);
            
            // ndofs.push_back(fe_sol.size());
            L2err[i].push_back(l2_err);
            H1serr[i].push_back(h1_serr);
        }
    }
    tabular_output(L2err, data_label, sol_name + "_L2err", output_folder);
    tabular_output(H1serr, data_label, sol_name + "_H1serr", output_folder);
}

void tabular_output(std::vector<std::vector<double>>& data, 
    std::vector<std::string>& data_label, const std::string& sol_name, 
    const std::string& output_folder) {
    
    std::cout << sol_name << std::endl;
    //Tabular output of the data
    // std::cout << std::left << std::setw(10) << data_label[i];
    std::cout << std::left;
    for(int i = 0; i < data_label.size(); ++i){
        std::cout << std::setw(10) << data_label[i];
    }
    std::cout << std::endl;
    std::cout << std::left << std::scientific << std::setprecision(2);
    for(int l = 0; l < data[0].size(); ++l) {
        std::cout << l << " & ";
        for(int i = 0; i < data.size(); ++i) {
            std::cout << std::setw(10) << data[i][l];
            if(i == data.size() - 1) {
                std::cout << " \\\\";
            } else {
                std::cout << " & ";
            }
        }
        std::cout << std::endl;
    }

    // write the result to the file
    std::string output_file = output_folder + sol_name + ".txt";
    std::ofstream out(output_file);

    out << data_label[0];
    for(int i = 1; i < data_label.size(); ++i) {
        out << " " << data_label[i];
    }
    out << std::endl;
    out << std::scientific << std::setprecision(2);
    for(int l = 0; l < data[0].size(); ++l) {
        out << l << " & ";
        for(int i = 0; i < data.size(); ++i) {
            if(i == data.size() - 1) {
                out << data[i][l] << " \\\\";
            } else {
                out << data[i][l] << " & ";
            }
        }
        out << std::endl;
    } 
}