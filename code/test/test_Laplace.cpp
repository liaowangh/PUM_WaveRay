#include <vector>
#include <functional>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/HE_solution.h"
#include "../utils/utils.h"

using namespace std::complex_literals;
using size_type = unsigned int;
using Scalar = std::complex<double>;
using coordinate_t = Eigen::Vector2d;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using triplet_t = Eigen::Triplet<Scalar>;
using SpMat_t = Eigen::SparseMatrix<Scalar>;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using FHandle_t = std::function<Scalar(const Eigen::Vector2d &)>;
using FunGradient_t = std::function<Eigen::Matrix<Scalar, 2, 1>(const coordinate_t&)>;
/*
 * Solve the Laplace equation 
 *  \Laplace u = 0 in \Omega
 *   u = g on \partial \Omega
 */
class Laplace {
public:
    Laplace(int levels, const std::string& mesh_path, FHandle_t g, int degree=20): L_(levels), g_(g), degree_(degree){
        auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
        reader_ = std::make_shared<lf::io::GmshReader>(std::move(mesh_factory), mesh_path);
        mesh_hierarchy_ = lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(reader_->mesh(), L_);
    }

    std::shared_ptr<lf::mesh::Mesh> getmesh(size_type l) { return mesh_hierarchy_->getMesh(l); }

    int nr_dofs(int l) {
        auto dofh = lf::assemble::UniformFEDofHandler(getmesh(l), {{lf::base::RefEl::kPoint(), 1}});
        return dofh.NumDofs();
    }

    std::pair<SpMat_t, Vec_t> build_equation(size_type l) {
        
        auto mesh = getmesh(l);  // get mesh
        auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
        
        auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});
        
        // assemble for <grad(u), grad(v)> 
        lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
        lf::mesh::utils::MeshFunctionConstant<double> mf_zero(0.0);
        lf::uscalfe::ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_zero)> 
            elmat_builder(fe_space, mf_identity, mf_zero);
        
        size_type N_dofs(dofh.NumDofs());
        lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);
        lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);
        
        // Assemble RHS vector
        Vec_t phi(N_dofs);
        phi.setZero();
        
        auto boundary_point{lf::mesh::utils::flagEntitiesOnBoundary(mesh, 2)};
        std::vector<std::pair<bool, Scalar>> ess_dof_select{};
        for(size_type dofnum = 0; dofnum < N_dofs; ++dofnum) {
            const lf::mesh::Entity &dof_node{dofh.Entity(dofnum)};
            const Eigen::Vector2d node_pos{lf::geometry::Corners(*dof_node.Geometry()).col(0)};
            const Scalar h_val = g_(node_pos);
            if(boundary_point(dof_node)) {
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
    
        return std::make_pair(A.makeSparse(), phi);
    }

    SpMat_t prolongation(int l){
        LF_ASSERT_MSG(l >= 0 && l < L_, "l in prolongation should be smaller to L");
        auto coarse_mesh = getmesh(l);
        auto fine_mesh   = getmesh(l+1);
        
        auto coarse_dofh = lf::assemble::UniformFEDofHandler(coarse_mesh, 
                            {{lf::base::RefEl::kPoint(), 1}});
        auto fine_dof    = lf::assemble::UniformFEDofHandler(fine_mesh, 
                            {{lf::base::RefEl::kPoint(), 1}});

        size_type n_c = coarse_dofh.NumDofs();
        size_type n_f = fine_dof.NumDofs();
        
        Mat_t M = Mat_t::Zero(n_c, n_f);

        for(const lf::mesh::Entity* edge: fine_mesh->Entities(1)) {
            nonstd::span<const lf::mesh::Entity* const> points = edge->SubEntities(1);
            size_type num_points = (*edge).RefEl().NumSubEntities(1); // number of endpoints, should be 2
            LF_ASSERT_MSG((num_points == 2), 
                "Every EDGE should have 2 kPoint subentities");
            for(int j = 0; j < num_points; ++j) {
                auto parent_p = mesh_hierarchy_->ParentEntity(l+1, *points[j]); // parent entity of current point 
                if(parent_p->RefEl() == lf::base::RefEl::kPoint()) {
                    // it's parent is also a NODE. If the point in finer mesh does not show in coarser mesh,
                    // then it's parent is an EDGE
                    M(coarse_mesh->Index(*parent_p), fine_mesh->Index(*points[j])) = 1.0;
                    M(coarse_mesh->Index(*parent_p), fine_mesh->Index(*points[1-j])) = 0.5;
                }
            }
        }
        return (M.transpose()).sparseView();
    }

    Vec_t solve(size_type l) {
        auto eq_pair = build_equation(l);
        Eigen::SparseLU<SpMat_t> solver;
        solver.compute(eq_pair.first);
        Vec_t fe_sol;
        if(solver.info() == Eigen::Success) {
            fe_sol = solver.solve(eq_pair.second);
        } else {
            LF_ASSERT_MSG(false, "Eigen Factorization failed");
        }
        return fe_sol;
    }

    std::vector<double> mesh_width() {
        std::vector<double> width(L_+1); 
        auto mesh = getmesh(0);
        double h0 = 0.0;

        for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
            const lf::geometry::Geometry *geo_ptr = cell->Geometry();

            // coordiante matrix: 2x3
            auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
            for(int i = 0; i < vertices.cols(); ++i) {
                for(int j = i + 1; j < vertices.cols(); ++j) {
                    double tmp = (vertices.col(i) - vertices.col(j)).norm();
                    if(tmp > h0) {
                        h0 = tmp;
                    }
                }
            }
        }
        width[0] = h0;
        for(int i = 1; i <= L_; ++i) {
            // because meshes are generated by uniform regular refinement
            width[i] = width[i-1] / 2;
        }
        return width;
    }

    double L2_Err(size_type l, const Vec_t& mu, const FHandle_t& u){
        auto mesh = getmesh(l);
        auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
        double res = 0.0;

        auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});

        for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
            const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        
            auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
            Eigen::Matrix3d X, tmp;
            tmp.block<3,1>(0,0) = Eigen::Vector3d::Ones();
            tmp.block<3,2>(0,1) = vertices.transpose();
            X = tmp.inverse();

            const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*cell));
            nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};

            auto integrand = [&X, &mu, &dofarray, &u](const Eigen::Vector2d& x)->Scalar {
                Scalar uh = 0.0;
                for(int i = 0; i < 3; ++i) {
                    uh += mu(dofarray[i]) * (X(0,i) + x.dot(X.block<2,1>(1,i)));
                }
                return std::abs((uh - u(x)) * (uh - u(x)));
            };
            res += std::abs(LocalIntegral(*cell, degree_, integrand));
        }
        return std::sqrt(res);
    }

    double H1_semiErr(size_type l, const Vec_t& mu, const FunGradient_t& grad_u) {
        auto mesh = getmesh(l);
        auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
        double res = 0.0;

        auto dofh = lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), 1}});

        for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
            const lf::geometry::Geometry *geo_ptr = cell->Geometry();
        
            auto vertices = geo_ptr->Global(cell->RefEl().NodeCoords());
            Eigen::Matrix3d X, tmp;
            tmp.block<3,1>(0,0) = Eigen::Vector3d::Ones();
            tmp.block<3,2>(0,1) = vertices.transpose();
            X = tmp.inverse();

            const lf::assemble::size_type no_dofs(dofh.NumLocalDofs(*cell));
            nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dofh.GlobalDofIndices(*cell)};

            auto grad_uh = (Eigen::Matrix<Scalar,2,1>() << 0.0, 0.0).finished();
            for(size_type i = 0; i < no_dofs; ++i) {
                grad_uh += mu(dofarray[i]) * X.block<2,1>(1,i);
            }
            // construct ||grad uh - grad u||^2_{cell}
            auto integrand = [&grad_uh, &grad_u](const Eigen::Vector2d& x)->Scalar {
                return std::abs((grad_uh - grad_u(x)).dot(grad_uh - grad_u(x)));
            };
            res += std::abs(LocalIntegral(*cell, degree_, integrand));
        }
        return std::sqrt(res);
    }

    Vec_t solve_multigrid(Vec_t& initial, size_type start_layer, int num_coarserlayer, int mu1, int mu2, bool solve_coarest=true) {
        LF_ASSERT_MSG((num_coarserlayer <= start_layer), 
            "please use a smaller number of wave layers");
        auto eq_pair = build_equation(start_layer);
        SpMat_t A = eq_pair.first;

        std::vector<SpMat_t> Op(num_coarserlayer + 1), prolongation_op(num_coarserlayer);
        std::vector<int> stride(num_coarserlayer + 1, 1);
        Op[num_coarserlayer] = A;
        for(int i = num_coarserlayer - 1; i >= 0; --i) {
            int idx = start_layer + i - num_coarserlayer;
            prolongation_op[i] = prolongation(idx);
            Op[i] = prolongation_op[i].transpose() * Op[i+1] * prolongation_op[i];
            // auto tmp = build_equation(idx);
            // Op[i] = tmp.first.makeSparse();
        }

        v_cycle(initial, eq_pair.second, Op, prolongation_op, stride, mu1, mu2, solve_coarest);
        return initial;
    }
private:
    int L_; // number of refinement steps

    std::shared_ptr<lf::io::GmshReader> reader_; // read the coarest mesh
    std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy_;
    int degree_;
    FHandle_t g_;
};

void Laplace_resolution(Laplace& laplace, int L, std::string& sol_name, std::string& output_folder,
    FHandle_t u, FunGradient_t grad_u) {
    std::vector<double> mesh_width = laplace.mesh_width();
    std::vector<double> L2err, H1serr, H1err;

    for(size_type level = 0; level <= L; ++level) {
        auto fe_sol = laplace.solve(level);
        
        double l2_err = laplace.L2_Err(level, fe_sol, u);
        double h1_serr = laplace.H1_semiErr(level, fe_sol, grad_u);
        
        L2err.push_back(l2_err);
        H1serr.push_back(h1_serr);
    }
    
    std::vector<std::vector<double>> err_data{mesh_width, L2err, H1serr};
    std::vector<std::string> data_label{"h", "L2_err", "H1_serr"};
    print_save_error(err_data, data_label, sol_name, output_folder);
}

int main() {
    std::string square = "../meshes/square.msh";
    std::string square_output = "../result_square/Laplace/";
    int L = 5; // refinement steps
    
    Harmonic_fun sol; 
    FHandle_t u = sol.get_fun();  // u(x,y) = exp(x+iy)
    FunGradient_t grad_u = sol.get_gradient();  
    std::string sol_name = "Harmonic";
    Laplace laplace(L, square, u);
    // Laplace_resolution(laplace, L, sol_name, square_output, u, grad_u);

    auto zero_fun = [](const coordinate_t& x) -> Scalar { return 0.0; };

    int N = laplace.nr_dofs(L);
    Vec_t v = Vec_t::Random(N); // initial value
    Vec_t uh = laplace.solve(L);

    int nr_coaremesh = 3;
    int nu1 = 3, nu2 = 3;

    for(int i = 0; i < 20; ++i) {
        std::cout << laplace.L2_Err(L, v - uh, zero_fun) << " ";
        std::cout << laplace.L2_Err(L, v, u) << std::endl;
        laplace.solve_multigrid(v, L, nr_coaremesh, nu1, nu2, false);
    }
    std::cout << "||u-uh||_2 = " << laplace.L2_Err(L, uh, u) << std::endl;
}

