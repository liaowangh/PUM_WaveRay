#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "HE_PUM.h"
#include "../utils/HE_solution.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

int main(){
    boost::filesystem::path here = __FILE__;
    auto mesh_path = (here.parent_path().parent_path() / ("meshes/coarest_mesh.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/square.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/tri2.msh")).string(); 
    std::string output_folder = "../plot_err/planwave_PUM/";
    size_type L = 3; // refinement steps
    double k = 5; // wave number
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;

    std::vector<int> num_waves(L+1, 3);
    
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);

    std::vector<std::string> sol_name{"h_plan_wave", "h_fundamental_sol", "h_spherical_wave"};
    
    /**** p-version, same mesh with increasing number of plan waves ****/
    /*
    std::vector<int> candidate_numwaves{3, 5, 7, 9};
    for(int i = 0; i < solutions.size(); ++i) {
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();

        std::vector<int> ndofs;
        std::vector<double> L2err, H1serr, H1err;

        for(int j = 0; j < candidate_numwaves.size(); ++j) {
            std::vector<int> tmp(L+1, candidate_numwaves[j]);
            HE_PUM he_pum(L, k, mesh_path, g, u, tmp, true);

            auto fe_sol = he_pum.solve(L);

            double l2_err = he_pum.L2_Err(L, fe_sol, u);
            double h1_serr = he_pum.H1_semiErr(L, fe_sol, grad_u);
            double h1_err = std::sqrt(l2_err*l2_err + h1_serr*h1_serr);
            
            ndofs.push_back(fe_sol.size());
            L2err.push_back(l2_err);
            H1serr.push_back(h1_serr);
            H1err.push_back(h1_err);
        }
        std::vector<std::vector<double>> err_data{L2err, H1err, H1serr};
        std::vector<std::string> err_str{"L2_err", "H1_err", "H1_serr"};
        print_save_error(ndofs, err_data, err_str, "p_" + sol_name[i], output_folder);
    } 
    */
    /*******************************************************************/
    
    /**** h-version, same number of plan waves with a set of meshes****/
    for(int i = 0; i < solutions.size(); ++i) {
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();
        HE_PUM he_pum(L, k, mesh_path, g, u, true, num_waves);
        test_solve(he_pum, sol_name[i], output_folder, L, u, grad_u);
    }
}