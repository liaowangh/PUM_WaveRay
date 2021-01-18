#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "PUM_WaveRay.h"

using namespace std::complex_literals;

lf::assemble::UniformFEDofHandler PUM_WaveRay::get_dofh(unsigned int level) {
    auto mesh = mesh_hierarchy->getMesh(level);
    size_type num = level == L ? 1 : std::pow(2, L + 1 - level);
    return lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), num}});
}

void PUM_WaveRay::Prolongation_LF(){
    P_LF = std::vector<Mat_t>(L);
    for(int l = 0; l < L; ++l) {
        auto coarse_mesh = mesh_hierarchy->getMesh(l);
        auto fine_mesh = mesh_hierarchy->getMesh(l+1);
        
        auto coarse_dofh = lf::assemble::UniformFEDofHandler(coarse_mesh,{{lf::base::RefEl::kPoint(), 1}});
        auto fine_dof = lf::assemble::UniformFEDofHandler(fine_mesh,{{lf::base::RefEl::kPoint(), 1}});
        
        size_type n_c = coarse_dofh.NumDofs();
        size_type n_f = fine_dof.NumDofs();
        
        Mat_t M(n_c, n_f);
        
        for(const lf::mesh::Entity* edge: fine_mesh->Entities(1)) {
            nonstd::span<const lf::mesh::Entity* const> points = edge->SubEntities(1);
            size_type num_points = (*edge).RefEl().NumSubEntities(1); // number of endpoints, should be 2
            for(int j = 0; j < num_points; ++j) {
                auto parent_p = mesh_hierarchy->ParentEntity(l+1, *points[j]); // parent entity of current point 
                if(parent_p->RefEl() == lf::base::RefEl::kPoint()) {
                    // it's parent is also a NODE, if the point in finer mesh does not show in coarser mesh,
                    // then it's parent is an EDGE
                    M(coarse_mesh->Index(*parent_p), fine_mesh->Index(*points[j])) = 1;
                    M(coarse_mesh->Index(*parent_p), fine_mesh->Index(*points[1-j])) = 0.5;
                }
            }
        }
        P_LF[l] = M.transpose();
    }
}

PUM_WaveRay::Scalar PUM_WaveRay::integration_mesh(int level, PUM_WaveRay::FHandle_t f) {
    lf::mesh::utils::MeshFunctionGlobal mf{f};
    // mf integrated over the mesh level
    Scalar res = lf::uscalfe::IntegrateMeshFunction(*(mesh_hierarchy->getMesh(level)), mf, 5);
    // traverse all triangles
//    for(auto e: mesh_hierarchy->getMesh(level)->Entities(0)){
//        Eigen::MatrixXd corners = lf::geometry::Corners(*(e->Geometry()));
//        double area = lf::geometry::Volume(*(e->Geometry()));
//        Scalar tmp = 0;
//        res += (f(corners.col(0)) + f(corners.col(1)) + f(corners.col(2))) * area / 3.;
//    }
    return res;
}

std::vector<double> generate_fre(int L, double k, int l, int t) {
    int N = (1 << (L + 1 - l));
    double d1 = std::cos(2*M_PI*t / N);
    double d2 = std::cos(2*M_PI*t / N);
    return {d1, d2};
}

void PUM_WaveRay::Prolongation_PW() {
    P_PW = std::vector<Mat_t>(L - 1);
    
    auto identity = [](Eigen::Vector2d x)->Scalar{ return 1.0;};
    
    for(int l = 0; l < L - 1; ++l) {
        int N1 = std::pow(2, L + 1 - l);
        int N2 = std::pow(2, L - l);
        
        Mat_t M(N2, N1);
        for(int i = 0; i < N2; ++i) {
            
            auto dl1t = generate_fre(L, k, l, i);
            auto dl1t1 = generate_fre(L, k, l, i+1);
            auto dlt = generate_fre(L, k, l, 2*i+1);
            
            Eigen::Matrix<Scalar, 2, 2> A;
            A << integration_mesh(L, identity),
                integration_mesh(L, exp_wave(dl1t1[0] - dl1t[0], dl1t1[1] - dl1t[1])),
                integration_mesh(L, exp_wave(dl1t[0] - dl1t1[0], dl1t[1] - dl1t1[1])),
                integration_mesh(L, identity);
            Eigen::Matrix<Scalar, 2, 1> b;
            b << integration_mesh(L, exp_wave(dlt[0] - dl1t[0], dlt[1] - dl1t[1])),
                integration_mesh(L, exp_wave(dlt[0] - dl1t1[0], dlt[1] - dl1t1[1]));
            auto tmp = A.colPivHouseholderQr().solve(b);
            
            M(i, 2 * i) = 1;
            M(i, 2 * i + 1) = tmp(0);
            M((i+1) / N2, 2 * i + 1) = tmp(1);
        }
        P_PW[l] = M;
    }
}

void PUM_WaveRay::Restriction_PW() {
    R_PW = std::vector<Mat_t>(L - 1);
    int N1 = 2;
    int N2 = 4;
    for(int l = L - 1; l > 0; --l) {
        N1 = N2;
        N2 = 2*N1;
        R_PW[l-1] = Mat_t::Zero(N2, N1);
        for(int i = 0; i < N1; ++i){
            R_PW[l-1](2*i, i) = 1;
        }
    }
}

/*
 * Return the linear operator that map a function from level l to level l+1 
 */
PUM_WaveRay::Mat_t PUM_WaveRay::Prolongation_PUM(int l) {
    LF_ASSERT_MSG((l < L), 
        "in prolongation, level should smaller than" << L);
    if(l == L - 1) {
        auto finest_mesh = mesh_hierarchy->getMesh(L);
        auto coarser_mesh = mesh_hierarchy->getMesh(L - 1);

        auto fine_dof = lf::assemble::UniformFEDofHandler(finest_mesh, {{lf::base::RefEl::kPoint(), 1}});
        auto coarser_dof = lf::assemble::UniformFEDofHandler(coarser_mesh, {{lf::base::RefEl::kPoint(), 1}});

        size_type n_f = fine_dof.NumDofs();
        size_type n_c = coarser_dof.NumDofs();

        size_type n_wave = 4;

        Mat_t res(n_f, n_wave * n_c);
        for(size_type t = 0; t < n_wave; ++t) {
            Mat_t P_L = P_LF[l]; // shape: n_f * n_c
            //
            // std::cout << P_L.rows() << " " << P_L.cols() << std::endl;
            // std::cout << n_f << " " << n_c << " " << std::endl;
            //
            auto fre_t = generate_fre(L, k, l, t);
            auto exp_t = exp_wave(fre_t[0], fre_t[1]);
            Vec_t exp_at_v(n_f);
            for(size_type i = 0; i < n_f; ++i) {
                const lf::mesh::Entity& v = fine_dof.Entity(i); // the entity to which i-th global shape function is associated
                Eigen::Vector2d v_coordinate = lf::geometry::Corners(*v.Geometry()).col(0);
                exp_at_v(i) = exp_t(v_coordinate);
            }
            res.block(0, t * n_c, n_f, n_c) = exp_at_v.asDiagonal() * P_L; // column wise * or / is not supported
        }
        return res;
    } else {
        auto Q = P_PW[l];
        auto P = P_LF[l];
        size_type n1 = Q.rows(), n2 = P.rows();
        size_type m1 = Q.cols(), m2 = P.cols();
        Mat_t res(n1*n2, m1*m2);
        for(int i = 0; i < n1; ++i) {
            for(int j = 0; j < m1; ++j) {
                res.block(i*n2, j*m2, n2, m2) = Q(i, j) * P;
            }
        }
        return res;
    }
}

/*
 * Return the linear operator that map a function from level l+1 to level l
 */
PUM_WaveRay::Mat_t PUM_WaveRay::Restriction_PUM(int l) {
    LF_ASSERT_MSG((l < L), 
        "in prolongation, level should smaller than" << L);
    if(l == L - 1) {
        return Prolongation_PUM(l).transpose();
    } else {
        auto Q = R_PW[l];
        auto P = P_LF[l].transpose();
        size_type n1 = Q.rows(), n2 = P.rows();
        size_type m1 = Q.cols(), m2 = P.cols();
        Mat_t res(n1*n2, m1*m2);
        for(int i = 0; i < n1; ++i) {
            for(int j = 0; j < m1; ++j) {
                res.block(i*n2, j*m2, n2, m2) = Q(i, j) * P;
            }
        }
        return res;
    }
}


std::pair<lf::assemble::COOMatrix<PUM_WaveRay::Scalar>, PUM_WaveRay::Vec_t>
PUM_WaveRay::build_equation(size_type level) {
    
    auto mesh = mesh_hierarchy->getMesh(level);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});
    
    // assemble for <grad(u), grad(v)> - k^2 uv
    lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
    lf::mesh::utils::MeshFunctionConstant<double> mf_k(-1. * k * k);
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_k)> 
    	elmat_builder(fe_space, mf_identity, mf_k);
    
    size_type N_dofs(dofh.NumDofs());
    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    // assemble boundary edge matrix, -i*k*u*v over \Gamma_R (outer boundary)
    lf::mesh::utils::MeshFunctionConstant<Scalar> mf_ik(-1i * k);

    auto outer_boundary{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)}; 

    if(hole_exist){
        // there is a hole inside, need to distinguish between outer and inner boundar
        auto outer_nr = reader->PhysicalEntityName2Nr("outer_boundary");
        auto inner_nr = reader->PhysicalEntityName2Nr("inner_boundary");

        // modify it to classify inner and outer boundary
        for(const lf::mesh::Entity* edge: mesh->Entities(1)) {
            if(outer_boundary(*edge)) {
                // find a boundary edge, need to determine if it's outer boundary
                const lf::mesh::Entity* parent_edge = edge;
                for(int i = level; i > 0; --i) {
                    parent_edge = mesh_hierarchy->ParentEntity(i, *parent_edge);
                }
                if(reader->IsPhysicalEntity(*parent_edge, inner_nr)) {
                    // it is the inner boundary
                    outer_boundary(*edge) = false;
                }
            }
        }
    }
                                             
    lf::uscalfe::MassEdgeMatrixProvider<double, decltype(mf_ik), decltype(outer_boundary)>
    	edge_mat_builder(fe_space, mf_ik, outer_boundary);
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector, \int_{\Gamma_R} gvdS
    Vec_t phi(N_dofs);
    phi.setZero();
    lf::mesh::utils::MeshFunctionGlobal mf_g{g};
    lf::mesh::utils::MeshFunctionGlobal mf_h{h};
    lf::uscalfe::ScalarLoadEdgeVectorProvider<double, decltype(mf_g), decltype(outer_boundary)>
    	edgeVec_builder(fe_space, mf_g, outer_boundary);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);
    
    if(hole_exist) {
        // Treatment of Dirichlet boundary conditions h = u|_{\Gamma_D} (inner boundary condition)
        // flag all nodes on the boundary
        auto inner_point{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 2)};
        // flag all nodes on the inner boundary
        for(const lf::mesh::Entity* edge: mesh->Entities(1)) {
            if(outer_boundary(*edge)) {
                // mark the points associated with outer boundary edge to false
                for(const lf::mesh::Entity* subent: edge->SubEntities(1)) {
                    inner_point(*subent) = false;
                }
            }
        }
        // Set up predicate: Run through all global shape functions and check whether
        // they are associated with an entity on the boundary, store Dirichlet data.
        std::vector<std::pair<bool, Scalar>> ess_dof_select{};
        for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
            const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
            const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
            const Scalar h_val = h(node_pos);
            if(inner_point(dof_node)) {
                // Dof associated with a entity on the boundary: "essential dof"
                // The value of the dof should be set to the value of the function h
                // at the location of the node.
                ess_dof_select.push_back({true, h_val});
            } else {
                ess_dof_select.push_back({false, h_val});
            }
        }

        // modify linear system of equations
        lf::assemble::FixFlaggedSolutionCompAlt<Scalar>(
            [&ess_dof_select](size_type dof_idx)->std::pair<bool, Scalar> {
                return ess_dof_select[dof_idx];},
        A, phi);
    }
    return std::make_pair(A, phi);
}


template <typename mat_type>
void Gaussian_Seidel(mat_type& A, PUM_WaveRay::Vec_t& phi, PUM_WaveRay::Vec_t& u, int t) {
    // u: initial value; t: number of iterations
    int N = A.rows();
    for(int i = 0; i < t; ++i){
        for(int j = 0; j < N; ++j) {
            auto tmp = A.row(j).dot(u);
            u(j) = (phi(j) - tmp + u(j) * A(j,j)) / A(j,j);
        }
    }
}

void PUM_WaveRay::v_cycle(Vec_t& u, size_type mu1, size_type mu2) {
    auto eq_pair = build_equation(L); // Equaiton Ax=f in finest mesh
    // const Eigen::SparseMatrix<Scalar> A(eq_pair.first.makeSparse());
    Mat_t A(eq_pair.first.makeDense());
    Vec_t f = eq_pair.second;

    std::vector<Vec_t> initiaLvalue(L + 1);
    std::vector<Vec_t> rhs_vec(L + 1);
    // std::vector<Eigen::SparseMatrix<Scalar>> Op(L + 1);
    std::vector<Mat_t> Op(L + 1);
    std::vector<Mat_t> restriction(L), prolongation(L);
    Op[L] = A;
    initiaLvalue[L] = u;
    rhs_vec[L] = f;
    // build coarser mesh operator
    for(size_type i = L - 1; i >= 0; --i) {
        restriction[i] = Restriction_PUM(i);
        prolongation[i] = Prolongation_PUM(i);
        Op[i] = restriction[i] * Op[i+1] * prolongation[i];
    }
    // initial guess on coarser mesh are all zero
    for(size_type i = L - 1; i >= 0; --i) {
        initiaLvalue[i] = Vec_t::Zero(Op[i].rows());
    }
    Gaussian_Seidel(A, rhs_vec[L], initiaLvalue[L], mu1);  // relaxation mu1 times on finest mesh
    for(int i = L - 1; i > 0; --i) {
        rhs_vec[i] = restriction[i] * (rhs_vec[i+1] - Op[i+1] * initiaLvalue[i+1]);
        Gaussian_Seidel(Op[i], rhs_vec[i], initiaLvalue[i], mu1); 
    }
    rhs_vec[0] = restriction[0] * (rhs_vec[1] - Op[1] * initiaLvalue[1]);
    initiaLvalue[0] = Op[0].colPivHouseholderQr().solve(rhs_vec[0]);
    for(int i = 1; i <= L; ++i) {
        initiaLvalue[i] += prolongation[i] * initiaLvalue[i-1];
        Gaussian_Seidel(Op[i], rhs_vec[i], initiaLvalue[i], mu2);
    }
    u = initiaLvalue[L];
}
/*
double PUM_WaveRay::L2_norm(size_type l, const Vec_t& mu) {
    auto mesh = mesh_hierarchy->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    auto mu_conj = mu.conjugate();
    // create mesh function represented by coefficient vector
    auto mf_mu = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu);
    auto mf_mu_conj = lf::uscalfe::MeshFunctionFE<double, Scalar>(fe_space, mu);

    double res = std::abs(lf::uscalfe::IntegrateMeshFunction(*mesh, mf_mu * mf_mu_conj, 5));
    return std::sqrt(res);
}

double PUM_WaveRay::H1_norm(size_type l, const Vec_t& mu) {
    auto mesh = mesh_hierarchy->getMesh(l);

    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});

    int N_dofs = dofh.NumDofs();
    lf::assemble::COOMatrix<double> mass_matrix(N_dofs, N_dofs);
    
    lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_identity)>
        elmat_builder(fe_space, mf_identity, mf_identity);
    
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mass_matrix);
    
    const Eigen::SparseMatrix<double> mass_mat = mass_matrix.makeSparse();
    double res = std::abs(mu.dot(mass_mat * mu.conjugate()));
    return std::sqrt(res);
}
*/
PUM_WaveRay::Vec_t PUM_WaveRay::fun_in_vec(size_type l, const FHandle_t& f) {
    auto dofh = lf::assemble::UniformFEDofHandler(getmesh(l), {{lf::base::RefEl::kPoint(), 1}});
    size_type N_dofs(dofh.NumDofs());
    Vec_t res(N_dofs);
    for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
        const lf::mesh::Entity& dof_node{dofh.Entity(dofnum)};
        const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
        res(dofnum) = f(node_pos);
    }
    return res;
}