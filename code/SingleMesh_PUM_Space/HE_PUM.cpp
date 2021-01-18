#include <vector>
#include <functional>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "HE_PUM.h"
#include "PUM_ElementMatrix.h"
#include "PUM_EdgeMat.h"
#include "PUM_EdgeVector.h"
#include "PUM_ElemVector.h"
#include "../utils/utils.h"

using namespace std::complex_literals;

/*
 * Build equation A*x = phi in PUM spaces
 */
std::pair<lf::assemble::COOMatrix<HE_PUM::Scalar>, HE_PUM::Vec_t> 
HE_PUM::build_equation(size_type level) {
    auto mesh = mesh_hierarchy->getMesh(level);  // get mesh
    
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);

    size_type N_wave(num_planwaves[level]);
    auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), N_wave}});
    size_type N_dofs(dofh.NumDofs());

    // assemble for <grad(u), grad(v)> - k^2 <u,v>
    // (u, v) -> \int_K \alpha grad u grad v.conj + \gamma * u v.conj dx
    PUM_FEElementMatrix elmat_builder(N_wave, k, 1., -1. * k * k);

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
    
    auto outer_boundary{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 1)};
    if(hole_exist) {
        // assemble boundary edge matrix, -i*k*u*v over \Gamma_R (outer boundary)
        // first need to distinguish between outer and inner boundar
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

    // (u,v) -> \int_e gamma * u * v.conj dS
    PUM_EdgeMat edge_mat_builder(fe_space, outer_boundary, N_wave, k, -1i * k);                                  
    lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
           
    // Assemble RHS vector, \int_{\Gamma_R} gvdS
    Vec_t phi(N_dofs);
    phi.setZero();
    // l(v) = \int_e g * v.conj dS(x)
    PUM_EdgeVec edgeVec_builder(fe_space, outer_boundary, N_wave, k, g);
    lf::assemble::AssembleVectorLocally(1, dofh, edgeVec_builder, phi);

    return std::make_pair(A, phi);
}

/*
 * To get the vector representation of function f w.r.t the PUM space,
 * a L2 projection is used.
 * P2(f) = uh, such that ||f-uh|| is minimized, which is equivalent to 
 * find uh such that (uh, v) = (f,v) for all v in the PUM space
 */
HE_PUM::Vec_t HE_PUM::fun_in_vec(size_type l, const FHandle_t& f) {
    auto mesh = mesh_hierarchy->getMesh(l);  // get mesh
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    
    size_type N_wave(num_planwaves[l]);
    auto dofh = get_dofh(l);
    size_type N_dofs(dofh.NumDofs());

    // assemble for \int u * v.conj dx
    PUM_FEElementMatrix elmat_builder(N_wave, k, 0, 1.);

    lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
    lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);

    // assemble for \int (f,v) dx
    Vec_t phi(N_dofs);
    phi.setZero();
    PUM_ElemVec elvec_builder(fe_space, N_wave, k, f);
    lf::assemble::AssembleVectorLocally(0, dofh, elvec_builder, phi);

    const Eigen::SparseMatrix<Scalar> A_crs(A.makeSparse());

    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(A_crs);
    Vec_t res;
    if(solver.info() == Eigen::Success) {
        res = solver.solve(phi);
    } else {
        LF_ASSERT_MSG(false, "Eigen Factorization failed")
    }
    return res;
}

// quadrature based norm computation for ||uh-u||
double HE_PUM::L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u) {
    auto mesh = mesh_hierarchy->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

    size_type N_wave(num_planwaves[l]);
    auto dofh = get_dofh(l);

    for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
         const lf::geometry::Geometry *geo_ptr = cell->Geometry();
    
        // Matrix storing corner coordinates in its columns(2x3 in this case)
        auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
        
        // suppose that the barycentric coordinate functions have the form
        // \lambda_i = a + b1*x+b2*y, then
        // \lambda_i = X(0,i) + X(1,i)*x + X(2,i)*y
        // grad \lambda_i = [X(1,i), X(2,i)]^T
        // grad \lambda_1_2_3 = X.block<2,3>(1,0)
        Eigen::Matrix3d X, tmp;
        tmp.block<3,1>(0,0) = Eigen::Vector3d::Ones();
        tmp.block<3,2>(0,1) = vertices.transpose();
        X = tmp.inverse();

        // Number of shape functions covering current entity
        const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*cell));
        //obtain global indices of those shape functions
        nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};
        // then uh(FE solution) restricted in the cell is \sum_i \lambda_(i/N_wave)*exp(ik d_{i%N_wave} x) * mu(dofarray(i))
        // construct ||grad uh - grad u||^2_{cell}

        auto integrand = [this, &mu, &u, &X, &N_wave, &dofarray](const Eigen::Vector2d& x)->Scalar {
            Scalar val_uh = 0.0;
            Scalar val_u  = u(x);
            for(int i = 0; i < 3; ++i) {
                for(int t = 0; t < N_wave; ++t) {
                    Eigen::Matrix<Scalar, 2, 1> d;
                    double pi = std::acos(-1);
                    d << std::cos(2*pi*t/N_wave), std::sin(2*pi*t/N_wave);
                    val_uh += mu(dofarray[i*N_wave+t]) * (X(0,i) + X(1,i)*x(0) + X(2,i)*x(1)) * std::exp(1i*k*d.dot(x));
                }
            }
            return std::abs((val_uh-val_u)*(val_uh-val_u));
        };
        res += std::abs(LocalIntegral(*cell, 10, integrand));
    }
    return std::sqrt(res);
}

double HE_PUM::H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) {
    auto mesh = mesh_hierarchy->getMesh(l);
    auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
    double res = 0.0;

    size_type N_wave(num_planwaves[l]);
    auto dofh = get_dofh(l);

    for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
         const lf::geometry::Geometry *geo_ptr = cell->Geometry();
    
        // Matrix storing corner coordinates in its columns(2x3 in this case)
        auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
        
        // suppose that the barycentric coordinate functions have the form
        // \lambda_i = a + b1*x+b2*y, then
        // \lambda_i = X(0,i) + X(1,i)*x + X(2,i)*y
        // grad \lambda_i = [X(1,i), X(2,i)]^T
        // grad \lambda_1_2_3 = X.block<2,3>(1,0)
        Eigen::Matrix3d X, tmp;
        tmp.block<3,1>(0,0) = Eigen::Vector3d::Ones();
        tmp.block<3,2>(0,1) = vertices.transpose();
        X = tmp.inverse();

        // Number of shape functions covering current entity
        const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*cell));
        //obtain global indices of those shape functions
        nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};
        // then uh(FE solution) restricted in the cell is \sum_i \lambda_(i/N_wave)*exp(ik d_{i%N_wave} x) * mu(dofarray(i))
        // construct ||grad uh - grad u||^2_{cell}

        auto integrand = [this, &mu, &grad_u, &X, &N_wave, &dofarray](const Eigen::Vector2d& x)->Scalar {
            // for f(x) = \lambda_i * exp(ikdt x)
            // grad f(x) = exp(ikdt x) * beta_i + ik*exp(ikdt x) * \lambda_i * dt
            Eigen::Matrix<Scalar, 2, 1> val_grad_uh, val_grad_u;
            val_grad_uh << 0.0, 0.0;
            val_grad_u = grad_u(x);
            for(int i = 0; i < 3; ++i) {
                for(int t = 0; t < N_wave; ++t) {
                    Eigen::Matrix<Scalar, 2, 1> d;
                    Eigen::Vector2d beta;
                    double pi = std::acos(-1);
                    d << std::cos(2*pi*t/N_wave), std::sin(2*pi*t/N_wave);
                    beta << X(1,i), X(2,i);
                    double lambda = X(0,i) + beta.dot(x);
                    val_grad_uh += mu(dofarray[i*N_wave+t]) * std::exp(1i*k*d.dot(x)) * (beta + 1i*k*lambda*d);
                }
            }
            return std::abs((val_grad_uh - val_grad_u).dot((val_grad_uh - val_grad_u).conjugate()));
        };
        res += std::abs(LocalIntegral(*cell, 10, integrand));
    }
    return std::sqrt(res);
}

double HE_PUM::H1_Err(size_type l, const Vec_t& mu, const FHandle_t& u, const FunGradient_t& grad_u) {
    double l2err = L2_Err(l, mu, u);
    double h1serr = H1_semiErr(l, mu, grad_u);
    return std::sqrt(l2err*l2err + h1serr*h1serr);
}