#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/HE_solution.h"
#include "PUM_WaveRay.h"

using namespace std::complex_literals;

using Scalar = std::complex<double>;
using size_type = unsigned int;

int main(){
    // mesh path
    boost::filesystem::path here = __FILE__;
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/coarest_mesh.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/tri2.msh")).string(); 
    auto mesh_path = (here.parent_path().parent_path() / ("meshes/square.msh")).string(); 
    size_type L = 2; // refinement steps
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
    // solutions[0] = std::make_shared<plan_wave>(k, 1., 0);
    // solutions[1] = std::make_shared<plan_wave>(k, 0, 1.);
    // solutions[2] = std::make_shared<plan_wave>(k, -1., 0);
 
    std::vector<std::string> sol_name{"waveray_plan_wave", "waveray_fundamental_sol", "waveray_spherical_wave"};
    // std::vector<std::string> sol_name{"wave_0_4", "wave_1_4", "wave_2_4"};
    for(int i = 0; i < solutions.size(); ++i) {
        if(i > 0){
            continue;
        }
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();
        PUM_WaveRay he_waveray(L, k, mesh_path, g, u, false, num_planwaves);

        /***************************************************************/
        std::cout << "start prolongation_planwave" << std::endl;
        he_waveray.Prolongation_planwave();

        // for(int i = 0; i < he_waveray.P_planwave.size(); ++i){
        //     std::cout << "From plan wave space " << i << " space " << i+1 << std::endl;
        //     std::cout << he_waveray.P_planwave[i] << std::endl;
        // }

        /***************************************************************/
        std::cout << "start prolongation_Lagrange" << std::endl;
        he_waveray.Prolongation_Lagrange();

        // for(int i = 0; i < he_waveray.P_Lagrange.size(); ++i){
        //     std::cout << "From mesh " << i << " to mesh " << i+1 << std::endl;
        //     std::cout << he_waveray.P_Lagrange[i] << std::endl;
        // }
        
        /***************************************************************/
        std::cout << "start prolongation_SE" << std::endl;
        he_waveray.Prolongation_SE();

        /***************************************************************/
        std::cout << "start prolongation_SE_S" << std::endl;
        he_waveray.Prolongation_SE_S();
        /***************************************************************/

        int start_layer = 2;
        int num_wavelayer = 2;
        auto dofh = he_waveray.get_dofh(start_layer);

        PUM_WaveRay::Vec_t fem_sol(dofh.NumDofs());
        fem_sol.setZero();
        std::cout << "Start v cycle;" << std::endl;
        he_waveray.solve(fem_sol, num_wavelayer, 10, 10);

        std::cout << he_waveray.HE_LagrangeO1::L2_Err(start_layer, fem_sol, u) << std::endl;

        // std::cout << std::endl << fem_sol << std::endl;

        std::cout << std::endl;
    }
}