#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
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
    
    // the exact function
    auto u_sol = [](const Eigen::Vector2d& x) -> mat_scalar {
        return std::exp(1i*(3*x(0) + 4*x(1)));
    };
    size_type L = 6; // refinement steps
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

    PUM_FEM pum_fem(L, k, mesh_path, g, u_sol);
    Eigen::Matrix<mat_scalar, Eigen::Dynamic, 1> appro_vec;
    pum_fem.v_cycle(appro_vec, 5, 5);
    
    // find the true vector representation
    auto dofh = lf::assemble::UniformFEDofHandler(pum_fem.getmesh(L), {{lf::base::RefEl::kPoint(), 1}});
    size_type N_dofs(dofh.NumDofs());
    Eigen::Matrix<mat_scalar, Eigen::Dynamic, 1> true_vec(N_dofs);
    for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
        const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
        const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
        true_vec(dofnum) = u_sol(node_pos);
    }
    std::cout << (appro_vec - true_vec).norm() / true_vec.norm() << std::endl;
    
    /*
    std::vector<double> ndofs;
    std::vector<double> L2err;
    
    for(size_type level = 0; level <= L; ++level) {
        auto eq_pair = pum_fem.build_equation(level);

        // solve linear system of equationx A*x = \phi
        const Eigen::SparseMatrix<mat_scalar> A_crs(eq_pair.first.makeSparse());
        Eigen::SparseLU<Eigen::SparseMatrix<mat_scalar>> solver;
        solver.compute(A_crs);
        Eigen::Matrix<mat_scalar, Eigen::Dynamic, 1> appro_vec;
        if(solver.info() == Eigen::Success) {
            appro_vec = solver.solve(eq_pair.second);
        } else {
            LF_ASSERT_MSG(false, "Eigen Factorization failed")
        }

        // find the true vector representation
        auto dofh = lf::assemble::UniformFEDofHandler(pum_fem.getmesh(level), {{lf::base::RefEl::kPoint(), 1}});
        size_type N_dofs(dofh.NumDofs());
        ndofs.push_back(N_dofs);
        Eigen::Matrix<mat_scalar, Eigen::Dynamic, 1> true_vec(N_dofs);

        for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
            const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
            const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
            true_vec(dofnum) = u_sol(node_pos);
        }
//        std::cout << (appro_vec - true_vec).norm() / true_vec.norm() << std::endl;
        L2err.push_back((appro_vec - true_vec).norm() / true_vec.norm());
    }
    */
    
    /*
    auto finest_mesh = pum_fem.getmesh(L);
    pum_fem.Restriction_PW();
    pum_fem.Prolongation_LF();
    pum_fem.Prolongation_PW();

    for(int i = 0; i < L; ++i) {
        auto P = pum_fem.Prolongation_PUM(i);
        auto Q = pum_fem.Restriction_PUM(i);
        std::cout << P.rows() << " " << P.cols() << " ";
        std::cout << Q.rows() << " " << Q.cols();
        std::cout << std::endl;
    }
    */

    // Tabular output of the results
    /*
    std::cout << std::left << std::setw(10) << "N" << std::setw(20)
              << "L2 err" << std::endl;
    for (int l = 0; l <= L; ++l) {
      std::cout << std::left << std::setw(10) << ndofs[l] << std::setw(20)
                << L2err[l] << std::endl;
    }
    */
    // write to the file
    
    return 0;
}
