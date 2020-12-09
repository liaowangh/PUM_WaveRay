#include <cmath>
#include <functional>
#include <string>

#include <gtest/gtest.h>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../pum_wave_ray/pum_fem.h"

using namespace std::complex_literals;

using data_type = std::complex<double>;
using size_tyhpe = unsigned int;

TEST(DebuggingPUM, solve_on_finest) {
    // read coarest mesh
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const std::string mesh_path = "/home/liaowang/Documents/master-thesis/code/meshes/coarest_mesh.msh"; 
    
    // the exact function
    auto u_sol = [](const Eigen::Vector2d& x) -> data_type {
        return std::exp(1i*(3*x(0) + 4*x(1)));
    };
    size_type L = 3; // refinement steps
    double k = 5; // wave number
    // Neumann boundary conditions
    auto g = [](const Eigen::Vector2d& x) -> data_type {
        double x1 = x(0), y1 = x(1);
        data_type res = 0;
        data_type u = std::exp(1i*(3*x1 + 4*y1));
        if(y1 == 0 && x1 <= 1 && x1 >= 0) {
            res = -9i * u;
        } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
            res = -2i * u;
        } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
            res = -i * u;
        } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
            res = -8i * u;
        } else {
            res = u;
        }
        return res;
    };
    // define mesh function
    lf::mesh::utils::MeshFunctionGlobal mf_u{u_sol};
    lf::mesh::utils::MeshFunctionGlobal mf_g{g};

    PUM_FEM<decltype(mf_g), decltype(mf_u)> pum_fem(L, k, mesh_path, mf_g, mf_u);
    
    auto finest_mesh = pum_fem.getmesh(L);
    auto eq_pair = pum_fem.build_finest();

    // TODO: solve equation can compare to the true value
}
