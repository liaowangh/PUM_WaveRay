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

#include "local_impedance_solver.h"
#include "../utils/HE_solution.h"
#include "../utils/utils.h"
#include "../LagrangeO1/HE_LagrangeO1.h"

using namespace std::complex_literals;

int main(){
    std::string square = "../meshes/square.msh";
    size_type L = 4; // refinement steps
    double k = 2.0; // wave number
    plan_wave sol(k, 0.8, 0.6);

    auto u = sol.get_fun();
    auto g = sol.boundary_g();
    auto grad_u = sol.get_gradient();
    HE_LagrangeO1 he_O1(L, k, square, g, u, false, 50);
   
    std::shared_ptr<lf::mesh::Mesh> mesh(he_O1.getmesh(1));
    epum_vertex_patch_info epatch_info(k, 8, mesh, false, he_O1.innerBdy_selector(1));

    std::vector<std::vector<int>>& adjacent_vertex = epatch_info.adjacent_vertex_;
    std::vector<std::vector<int>>& adjacent_cell = epatch_info.adjacent_cell_;

    for(int i = 0; i < adjacent_vertex.size(); ++i) {
        // if(i > 0) break;
        std::cout << "Vertex " << i << std::endl;
        std::cout << "Index of vertices in vertex patch: ";
        for(auto j : adjacent_vertex[i]){
            std::cout << j << " ";
        }
        auto tmp = epatch_info.patch_idx_map(i);
        std::cout << "Q(" << tmp.first << "," << i << ") = " << i << std::endl;
        Eigen::MatrixXd Qi = Eigen::MatrixXd(tmp.second);
        std::cout << "Mapping matrix size: " << Qi.rows() << "x" << Qi.cols() << std::endl;
        auto Al_pair = epatch_info.localMatrix_idx(i);
        std::cout << "Local matrix size: " << Al_pair.first.rows() << "x" << Al_pair.first.cols() << std::endl;
        std::cout << "Center vertex local index: " << Al_pair.second << std::endl;
        // for(int row = 0; row < Qi.rows(); ++row) {
        //     for(int col = 0; col < Qi.cols(); ++col) {
        //         if(Qi(row, col) != 0 ) {
        //             std::cout << "Q[" << row << ", " << col << "] = 1" << std::endl;
        //         }
        //     }
        // }
        
        // std::cout << std::endl << "Index of cells in vertex patch: ";
        // for(auto j : adjacent_cell[i]) {
        //     std::cout << j << " ";
        // }
        // std::cout << std::endl;
    }
    // for(int i = 0; i < adjacent_cell.size(); ++i) {
    //     std::cout << "Vertex " << i << std::endl;
    //     std::cout << "Index of boundary edge: ";
    //     auto patch_bdy = epatch_info.patch_boundary_selector(i);
    //     for(const lf::mesh::Entity* edge:  mesh->Entities(1)) {
    //         if(patch_bdy(*edge)) {
    //             std::cout << mesh->Index(*edge) << " ";
    //         }
    //     }
    //     std::cout << std::endl;
    // }

    // // Edge index:
    // for(const lf::mesh::Entity* edge: mesh->Entities(1)) {
    //     std::cout << "Edge " << mesh->Index(*edge) << " [ ";
    //     for(const lf::mesh::Entity* node: edge->SubEntities(1)) {
    //         std::cout << mesh->Index(*node) << " ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
}