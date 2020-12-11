#include <cmath>
#include <functional>
#include <string>

#include <boost/filesystem.hpp>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "pum_fem.h"

using namespace std::complex_literals;

using mat_scalar = std::complex<double>;
using size_type = unsigned int;

int main(){
    // mesh path
    boost::filesystem::path here = __FILE__;
    auto mesh_path = (here.parent_path().parent_path() / ("meshes/coarest_mesh.msh")).string(); 

    // const std::string mesh_path = "/home/liaowang/Documents/master-thesis/code/meshes/coarest_mesh.msh"; 
    
    // the exact function
    auto u_sol = [](const Eigen::Vector2d& x) -> mat_scalar {
        return std::exp(1i*(3*x(0) + 4*x(1)));
    };
    size_type L = 3; // refinement steps
    double k = 5; // wave number
    // Neumann boundary conditions
    auto g = [](const Eigen::Vector2d& x) -> mat_scalar {
        double x1 = x(0), y1 = x(1);
        mat_scalar res = 0;
        mat_scalar u = std::exp(1i*(3*x1 + 4*y1));
        if(y1 == 0 && x1 <= 1 && x1 >= 0) {
            res = -9i * u;
        } else if(x1 == 1 && y1 >= 0 && y1 <= 1) {
            res = -2i * u;
        } else if(y1 == 1 && x1 >= 0 && x1 <= 1) {
            res = -1i * u;
        } else if(x1 == 0 && y1 >= 0 && y1 <= 1) {
            res = -8i * u;
        } else {
            res = u;
        }
        return res;
    };
    // define mesh function
    // lf::mesh::utils::MeshFunctionGlobal mf_u{u_sol};
    // lf::mesh::utils::MeshFunctionGlobal mf_g{g};

    PUM_FEM pum_fem(L, k, mesh_path, g, u_sol);
    
    auto finest_mesh = pum_fem.getmesh(L);
    pum_fem.Prolongation_Q();

    std::cout << pum_fem.Q[0].rows() << " " << pum_fem.Q[0].cols() << std::endl;
    // std::cout << pum_fem.Q[0] << std::endl;
    /*
    auto eq_pair = pum_fem.build_equation(L);

    // solve linear system of equationx A*x = \phi
    const Eigen::SparseMatrix<mat_scalar> A_crs(eq_pair.first().makeSparse());
    Eigen::SparseLU<Eigen::SparseMatrix<mat_scalar>> solver;
    solver.compute(A_crs);
    if(solver.info() == Eigen::Success) {
        auto appro_vec = solver.solve(eq_pair.second());
    } else {
        LF_ASSERT_MSG(flase, "Eigen Factorization failed")
    }

    // find the true vector representation
    auto dofh = lf::assemble:UniformFEDofHandler(finest_mesh, {{lf::base::RefEl::kPoint(), 1}});
    size_type N_dofs(dofh.NumDofs());
    Eigen::Matrix<mat_scalar, N_dofs, 1> true_vec;

    for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
        const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
        const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col()};
        true_vec(dofnum) = u_sol(node_pos);
    }
    std::cout << (appro_vec - u_sol).norm() << std::endl;
    */

    return 0;
}
