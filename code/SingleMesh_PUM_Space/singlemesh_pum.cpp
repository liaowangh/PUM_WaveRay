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
    auto mesh_path = (here.parent_path().parent_path() / ("meshes/coarest_mesh.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/no_hole.msh")).string(); 
    // auto mesh_path = (here.parent_path().parent_path() / ("meshes/tri2.msh")).string(); 
    size_type L = 3; // refinement steps
    double k = 2; // wave number
    Eigen::Vector2d c; // center in fundamental solution
    c << 10.0, 10.0;

    std::vector<int> num_waves(L+1, 4);
    
    std::vector<std::shared_ptr<HE_sol>> solutions(3);
    solutions[0] = std::make_shared<plan_wave>(k, 0.6, 0.8);
    solutions[1] = std::make_shared<fundamental_sol>(k, c);
    solutions[2] = std::make_shared<Spherical_wave>(k, 2);
    // solutions[0] = std::make_shared<plan_wave>(k, 1., 0);
    // solutions[1] = std::make_shared<plan_wave>(k, 0, 1.);
    // solutions[2] = std::make_shared<plan_wave>(k, -1., 0);
 
    std::vector<std::string> sol_name{"pum_plan_wave", "pum_fundamental_sol", "pum_spherical_wave"};
    // std::vector<std::string> sol_name{"wave_0_4", "wave_1_4", "wave_2_4"};
    for(int i = 0; i < solutions.size(); ++i) {
        // if(i > 0){
        //     continue;
        // }
        auto u = solutions[i]->get_fun();
        auto grad_u = solutions[i]->get_gradient();
        auto g = solutions[i]->boundary_g();
        HE_PUM he_pum(L, k, mesh_path, g, u, num_waves, true);
        // test_localcomputation(he_pum, 0, k);
        // auto vec_coeff = he_pum.fun_in_vec(0, u);
        // std::cout << std::left << std::setw(20) << he_pum.L2_Err(0, vec_coeff, u) << std::setw(20)
        //           << he_pum.H1_Err(0, vec_coeff, u, grad_u) << std::endl;
        // std::cout << vec_coeff << std::endl << std::endl;
        solve_directly(he_pum, sol_name[i], L, u, grad_u);
        std::cout << std::endl;
    }
}


/*
#include "PUM_ElementMatrix.h"
#include "PUM_EdgeMat.h"
#include "PUM_EdgeVector.h"
#include "PUM_ElemVector.h"

void test_localcomputation(HE_FEM& he_fem, size_type l, double k) {
    auto mesh = he_fem.getmesh(l);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    size_type N_wave = he_fem.Dofs_perNode(l);
    auto dofh = he_fem.get_dofh(l);
    size_type N_dofs(dofh.NumDofs());

    // (u, v) -> \int_K \alpha * (grad u, grad v) + \gamma * (u, v) dx
    PUM_FEElementMatrix elmat_builder(N_wave, k, 1., 0);

    lf::assemble::COOMatrix<Scalar> ElemMat(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, ElemMat);
    
    std::cout << "Element Matrix 1" << std::endl;
    std::cout << ElemMat.makeSparse() << std::endl;
    //******************************************************
    PUM_FEElementMatrix elmat_builder2(N_wave, k, 0, 1.);

    lf::assemble::COOMatrix<Scalar> ElemMat2(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder2, ElemMat2);
    
    std::cout << "Element Matrix 2" << std::endl;
    std::cout << ElemMat2.makeSparse() << std::endl;
    //******************************************************
    auto outer_boundary{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)};

    // (u,v) -> \int_e gamma * (u,v) dS
    PUM_EdgeMat edge_mat_builder(fe_space, outer_boundary, N_wave, k, 1.);  
    lf::assemble::COOMatrix<Scalar> EdgeMat(N_dofs, N_dofs);                                
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, EdgeMat);
        
    std::cout << "Edge Matrix" << std::endl;
    std::cout << EdgeMat.makeSparse() << std::endl;
    
    // Assemble RHS vector, \int_{\Gamma_R} gv.conj dS
    Vec_t phi(N_dofs);
    phi.setZero();
    // l(v) = \int_e g * v.conj dS(x)
    PUM_EdgeVec edgeVec_builder(fe_space, outer_boundary, N_wave, k, he_fem.g);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);

    std::cout << "RHS vector" << std::endl;
    std::cout << phi << std::endl;
}
*/