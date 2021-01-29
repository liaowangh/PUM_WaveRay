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
    // mesh path
    boost::filesystem::path here = __FILE__;
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/coarest_mesh.msh")).string(); 
    auto mesh_path = (here.parent_path().parent_path() / ("meshes/square.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/tri2.msh")).string(); 
    size_type L = 4; // refinement steps
    double k = 2; // wave number
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;

    std::vector<int> num_waves(L+1, 3);
    
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.8, 0.6);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);
    // solutions[0] = std::make_shared<plan_wave>(k, 1., 0);
    // solutions[1] = std::make_shared<plan_wave>(k, 0, 1.);
    // solutions[2] = std::make_shared<plan_wave>(k, -1., 0);
 
    std::vector<std::string> sol_name{"pum3_plan_wave", "pum3_fundamental_sol", "pum3_spherical_wave"};
    // std::vector<std::string> sol_name{"wave_0_4", "wave_1_4", "wave_2_4"};
    for(int i = 0; i < solutions.size(); ++i) {
        if(i > 0){
            continue;
        }
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();
        HE_PUM he_pum(L, k, mesh_path, g, u, num_waves, false);
        solve_directly(he_pum, sol_name[i], L, u, grad_u);
        std::cout << std::endl;
    }
}