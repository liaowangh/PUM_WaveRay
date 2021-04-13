#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <unordered_set>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../Pum_WaveRay/HE_FEM.h"
#include "../ExtendPum/ExtendPUM_ElementMatrix.h"
#include "../ExtendPum/ExtendPUM_EdgeMat.h"

using Scalar = std::complex<double>;
using Vec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using triplet_t = Eigen::Triplet<Scalar>;
using SpMat_t = Eigen::SparseMatrix<Scalar>;
using namespace std::complex_literals;


template <typename SCALAR, typename DIFF_COEFF, typename REACTION_COEFF>
class patch_ReactionDiffusionElementMatrixProvider: 
    public lf::uscalfe::ReactionDiffusionElementMatrixProvider<SCALAR, DIFF_COEFF, REACTION_COEFF> {
public:
    // constructor
    patch_ReactionDiffusionElementMatrixProvider(
        std::shared_ptr<const lf::uscalfe::UniformScalarFESpace<SCALAR>> fe_space,
        DIFF_COEFF alpha, REACTION_COEFF gamma,
        lf::mesh::utils::CodimMeshDataSet<bool> patch_element_selector):
        lf::uscalfe::ReactionDiffusionElementMatrixProvider<SCALAR, DIFF_COEFF, REACTION_COEFF>(fe_space, alpha, gamma),
        patch_element_selector_(patch_element_selector){}

    bool isActive(const lf::mesh::Entity & cell) override {
        return patch_element_selector_(cell); 
    }

private:
    lf::mesh::utils::CodimMeshDataSet<bool> patch_element_selector_;
};

class vertex_patch_info {
public:
    vertex_patch_info(double wave_number, std::shared_ptr<lf::mesh::Mesh> mesh, bool hole,
        lf::mesh::utils::CodimMeshDataSet<bool> inner_bdy): 
        k_(wave_number), mesh_(mesh), hole_(hole), inner_boundary_(inner_bdy) {
        int n = mesh->NumEntities(2); // number of vertices
        std::vector<std::unordered_set<int>> adjacent_vertex(n);
        std::vector<std::unordered_set<int>> adjacent_cell(n);
        for(const lf::mesh::Entity* cell: mesh->Entities(0)) {
            nonstd::span<const lf::mesh::Entity* const> points = cell->SubEntities(2);
            for(int i = 0; i < 3; ++i) {
                int p_idx = mesh->Index(*points[i]);
                adjacent_cell[p_idx].insert(mesh->Index(*cell));
                
                for(int j = 0; j < 3; ++j) {
                    adjacent_vertex[p_idx].insert(mesh->Index(*points[j]));
                }
            }
        }

        // now adjacent_vertex and adjacent_cell only contain one layer of cells and points
        // need to extend it to contain two layers
        auto adjacent_vertex_copy = adjacent_vertex;
        auto adjacent_cell_copy = adjacent_cell;
        for(int i = 0; i < n; ++i) {
            for(auto j: adjacent_vertex_copy[i]) {
                for(auto k: adjacent_vertex_copy[j]) {
                    adjacent_vertex[i].insert(k);
                }

                for(auto k: adjacent_cell_copy[j]){
                    adjacent_cell[i].insert(k);
                }
            }
        }
        // now ajdacent_vertex[i] contains index of vertices in vertex patch i
        // ajdacent_cell[i] contains index of cell in vertex patch i
        for(int i = 0; i < n; ++i) {
            adjacent_vertex_.push_back(std::vector<int>(adjacent_vertex[i].begin(), adjacent_vertex[i].end()));
            adjacent_cell_.push_back(std::vector<int>(adjacent_cell[i].begin(), adjacent_cell[i].end()));
        }
    }

    /*
    * iterative through all the edges of triangels in vetex patch
    * the edges appear once belong to the boundary
    * appear twice belong to the interior of the vertex patch
    */
    lf::mesh::utils::CodimMeshDataSet<bool> patch_boundary_selector(int l) {
        lf::mesh::utils::CodimMeshDataSet<bool> patch_bdy(mesh_, 1, false);
        for(int cell_idx: adjacent_cell_[l]) {
            const lf::mesh::Entity* cell = mesh_->EntityByIndex(0, cell_idx);
            for(const lf::mesh::Entity* edge: cell->SubEntities(1)) {
                if(!inner_boundary_(*edge)){
                    patch_bdy(*edge) = !patch_bdy(*edge);
                }
            }
        }
        return patch_bdy;
    }

    lf::mesh::utils::CodimMeshDataSet<bool> patch_element_selector(int l) {
        lf::mesh::utils::CodimMeshDataSet<bool> patch_element(mesh_, 0, false);
        for(int cell_idx: adjacent_cell_[l]) {
            const lf::mesh::Entity* cell = mesh_->EntityByIndex(0, cell_idx);
            patch_element(*cell) = true;
        }
        return patch_element;
    }

    virtual std::pair<int, Eigen::SparseMatrix<double>> patch_idx_map(int l) {
        int center_vertex_local_idx; // 
        int N = mesh_->NumEntities(2); // number of global basis functions
        int n = adjacent_vertex_[l].size(); // number of basis functions in vertex patch l
        Eigen::SparseMatrix<double> Q(n, N);

        std::vector<int>& adj_vertex = adjacent_vertex_[l];
        std::vector<Eigen::Triplet<double>> triplets;
        for(int i = 0; i < adj_vertex.size(); ++i) {
            if(adj_vertex[i] == l) {
                center_vertex_local_idx = i;
            }
            triplets.push_back(Eigen::Triplet<double>(i, adj_vertex[i], 1.0));
        }
        Q.setFromTriplets(triplets.begin(), triplets.end());
        return {center_vertex_local_idx, Q};
    }

    virtual std::pair<Mat_t, int> localMatrix_idx(int l) {
        auto patch_bdy = patch_boundary_selector(l);
        auto patch_element = patch_element_selector(l);
        auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_);
        lf::assemble::UniformFEDofHandler dofh(mesh_, {{lf::base::RefEl::kPoint(), 1}});
        int N_dofs(dofh.NumDofs());
        lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);

        // assembel for <grad u, grad v> - k^2 uv over vertex patch space l
        lf::mesh::utils::MeshFunctionConstant<double> mf_identity(1.);
        lf::mesh::utils::MeshFunctionConstant<double> mf_k(-1. * k_ * k_);
        patch_ReactionDiffusionElementMatrixProvider<double, decltype(mf_identity), decltype(mf_k)> 
            elmat_builder(fe_space, mf_identity, mf_k, patch_element);
        lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);

        // assemble for <u, v> over boundary of patch space withoud the inner boundary
        lf::mesh::utils::MeshFunctionConstant<Scalar> mf_ik(-1i * k_);
        lf::uscalfe::MassEdgeMatrixProvider<double, decltype(mf_ik), decltype(patch_bdy)>
    	    edge_mat_builder(fe_space, mf_ik, patch_bdy);
        lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
        
        auto index_info = patch_idx_map(l);

        auto Q = index_info.second;
        Mat_t Al = Q * A.makeSparse() * Q.transpose();
        return {Al, index_info.first};
    }

    virtual void relaxation(Vec_t& v, Vec_t& r) {
        for(int l = 0; l < v.size(); ++l) {
            auto tmp = localMatrix_idx(l);
            Mat_t Al = tmp.first;       
            Vec_t local_residual = Vec_t::Zero(Al.rows());
            local_residual(tmp.second) = r(l);
        
            Vec_t el = Al.colPivHouseholderQr().solve(local_residual);
            v(l) += el(tmp.second);
        }
    }
public:
    double k_;
    bool hole_;
    std::shared_ptr<lf::mesh::Mesh> mesh_;
    std::vector<std::vector<int>> adjacent_vertex_;
    std::vector<std::vector<int>> adjacent_cell_;
    lf::mesh::utils::CodimMeshDataSet<bool> inner_boundary_;
};

class patch_ExtendPUM_ElementMatrix: public ExtendPUM_ElementMatrix {
public:
    // constructor
    patch_ExtendPUM_ElementMatrix(size_type N, double k, Scalar alpha, Scalar gamma,
        lf::mesh::utils::CodimMeshDataSet<bool> patch_element_selector, int degree=20):
        patch_element_selector_(patch_element_selector),
        ExtendPUM_ElementMatrix(N, k, alpha, gamma, degree){}

    bool isActive(const lf::mesh::Entity & cell) override {
         return patch_element_selector_(cell); 
    }
private:
    lf::mesh::utils::CodimMeshDataSet<bool> patch_element_selector_;
};

class epum_vertex_patch_info: public vertex_patch_info {
public:
    epum_vertex_patch_info(double wave_number, int nr_waves, std::shared_ptr<lf::mesh::Mesh> mesh, 
        bool hole, lf::mesh::utils::CodimMeshDataSet<bool> inner_bdy, int degree=30): 
        N_wave(nr_waves), quad_degree(degree),
        vertex_patch_info(wave_number, mesh, hole, inner_bdy) {}

    std::pair<int, Eigen::SparseMatrix<double>> patch_idx_map(int l) override {
        int center_vertex_local_idx; // 
        int N_nodal = mesh_->NumEntities(2); // number of global nodal basis functions
        int n = adjacent_vertex_[l].size(); // number of nodal basis functions in vertex patch
        Eigen::SparseMatrix<double> Q(n * (N_wave + 1), N_nodal * (N_wave + 1));

        std::vector<int>& adj_vertex = adjacent_vertex_[l];
        std::vector<Eigen::Triplet<double>> triplets;
        for(int i = 0; i < adj_vertex.size(); ++i) {
            if(adj_vertex[i] == l) {
                center_vertex_local_idx = i;
            }
            for(int t = 0; t <= N_wave; ++t) {
                triplets.push_back(Eigen::Triplet<double>(i*(N_wave+1)+t, adj_vertex[i]*(N_wave+1)+t, 1.0));
            }
        }
        Q.setFromTriplets(triplets.begin(), triplets.end());
        return {center_vertex_local_idx, Q};
    }

    std::pair<Mat_t, int> localMatrix_idx(int l) override {
        auto patch_bdy = patch_boundary_selector(l);
        auto patch_element = patch_element_selector(l);
        auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_);
        lf::assemble::UniformFEDofHandler dofh(mesh_, {{lf::base::RefEl::kPoint(), N_wave+1}});
        int N_dofs(dofh.NumDofs());
        lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);

        // assembel for <grad u, grad v> - k^2 uv over vertex patch space l
        patch_ExtendPUM_ElementMatrix elmat_builder(N_wave, k_, 1.0, -k_ * k_, patch_element, quad_degree);
        lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);

        // assemble for <u, v> over boundary of patch space withoud the inner boundary
        ExtendPUM_EdgeMat edge_mat_builder(fe_space, patch_bdy, N_wave, k_, 1.0, quad_degree);
        lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edge_mat_builder, A);
        
        auto index_info = patch_idx_map(l);

        auto Q = index_info.second;
        Mat_t Al = Q * A.makeSparse() * Q.transpose();
        return {Al, index_info.first};
    }

    void relaxation(Vec_t& v, Vec_t& r) override {
        for(int l = 0; l < v.size() / (N_wave + 1); ++l) {
            auto tmp = localMatrix_idx(l);
            int local_idx = tmp.second;
            Mat_t Al = tmp.first;       
            Vec_t local_residual = Vec_t::Zero(Al.rows());
            local_residual.segment(local_idx*(N_wave+1), N_wave+1) = r.segment(l*(N_wave+1), N_wave+1);
            Vec_t el = Al.colPivHouseholderQr().solve(local_residual);
            v.segment(l*(N_wave+1), N_wave+1) += el.segment(local_idx*(N_wave+1), N_wave+1);
        }
    }
public:
    int N_wave; // number of plan waves
    int quad_degree;
};

/************ smoothing element ****************/

class impedance_smoothing_element {
public:
    impedance_smoothing_element(HE_FEM& he_fem, int L, int nr_coarselayers, 
        double k, double kh_threshold): k_(k), kh_threshold_(kh_threshold){
        
        auto eq_pair = he_fem.build_equation(L);
        SpMat_t A(eq_pair.first.makeSparse());
        auto mesh_width = he_fem.mesh_width();
        mw = std::vector<double>(nr_coarselayers + 1);
        I = std::vector<SpMat_t>(nr_coarselayers);
        Op = std::vector<SpMat_t>(nr_coarselayers + 1);
        Op[nr_coarselayers] = A;
        mw[nr_coarselayers] = mesh_width[L];
        for(int i = nr_coarselayers - 1; i >= 0; --i) {
            int idx = L + i - nr_coarselayers;
            I[i] = he_fem.prolongation(idx);
            Op[i] = I[i].transpose() * Op[i+1] * I[i];
            // auto tmp = he_fem.build_equation(idx);
            // Op[i] = tmp.first.makeSparse();
            mw[i] = mesh_width[idx];
        }

        for(int i = nr_coarselayers; i >= 0; --i) {
            int idx = L + i - nr_coarselayers;
            if(k_ * mw[i] >= kh_threshold_) {
                int n = Op[i].rows();
                impedance_matrix[i] = std::vector<Mat_t>(n);
                impedance_idx[i] = std::vector<int>(n);
                vertex_patch_info patch(k_, he_fem.getmesh(idx), he_fem.hole_exist_, he_fem.innerBdy_selector(idx));
                for(int l = 0; l < n; ++l) {
                    auto local_info_pair = patch.localMatrix_idx(l);
                    impedance_matrix[i][l] = local_info_pair.first;
                    impedance_idx[i][l] = local_info_pair.second;
                }
            }
        }        
    }

    void smoothing(int l, Vec_t& u, Vec_t& rhs) {
        if(mw[l] * k_ < kh_threshold_) {
            Gaussian_Seidel(Op[l], rhs, u, 1, 3);
        } else {
            for(int i = 0; i < u.size(); ++i) {
                Mat_t Al = impedance_matrix[l][i];
                int local_idx = impedance_idx[l][i];
                Vec_t local_residual = Vec_t::Zero(Al.rows());
                local_residual(local_idx) = rhs(i);
                Vec_t el = Al.colPivHouseholderQr().solve(local_residual);
                u(i) += el(local_idx);
            }
        }
    }

public:
    double kh_threshold_;
    double k_;
    std::vector<double> mw;  // mesh_width
    std::vector<SpMat_t> Op;
    std::vector<SpMat_t> I;   // prolongation operator
    std::unordered_map<int, std::vector<Mat_t>> impedance_matrix;
    std::unordered_map<int, std::vector<int>> impedance_idx;
};

class epum_impedance_smoothing_element {
public:
    epum_impedance_smoothing_element(HE_FEM& he_fem, int L, int nr_coarselayers, 
        double k, double kh_threshold): k_(k), kh_threshold_(kh_threshold) {
        
        auto eq_pair = he_fem.build_equation(L);
        SpMat_t A(eq_pair.first.makeSparse());
        auto mesh_width = he_fem.mesh_width();
        mw = std::vector<double>(nr_coarselayers + 1);
        I = std::vector<SpMat_t>(nr_coarselayers);
        Op = std::vector<SpMat_t>(nr_coarselayers + 1);
        Op[nr_coarselayers] = A;
        mw[nr_coarselayers] = mesh_width[L];
        for(int i = nr_coarselayers - 1; i >= 0; --i) {
            int idx = L + i - nr_coarselayers;
            I[i] = he_fem.prolongation(idx);
            Op[i] = I[i].transpose() * Op[i+1] * I[i];
            // auto tmp = he_fem.build_equation(idx);
            // Op[i] = tmp.first.makeSparse();
            mw[i] = mesh_width[idx];
        }

        dofs_perNode = std::vector<int>(nr_coarselayers + 1);
        for(int i = nr_coarselayers; i >= 0; --i) {
            int idx = L + i - nr_coarselayers;
            dofs_perNode[i] = he_fem.Dofs_perNode(idx);
            if(k_ * mw[i] < kh_threshold_) {
                continue;
            }
            if(i == nr_coarselayers) {
                // still the Lagrangian finite element space
                int n = Op[i].rows();
                impedance_matrix[i] = std::vector<Mat_t>(n);
                impedance_idx[i] = std::vector<int>(n);
                vertex_patch_info patch(k_, he_fem.getmesh(idx), he_fem.hole_exist_, he_fem.innerBdy_selector(idx));
                for(int l = 0; l < n; ++l) {
                    auto local_info_pair = patch.localMatrix_idx(l);
                    impedance_matrix[i][l] = local_info_pair.first;
                    impedance_idx[i][l] = local_info_pair.second;
                }
            } 
            else {
                // the extend PUM space
                int n = Op[i].rows() / dofs_perNode[i]; // number of nodes
                impedance_matrix[i] = std::vector<Mat_t>(n);
                impedance_idx[i] = std::vector<int>(n);
                epum_vertex_patch_info patch(k_, dofs_perNode[i]-1, he_fem.getmesh(idx), 
                    he_fem.hole_exist_, he_fem.innerBdy_selector(idx));
                for(int l = 0; l < n; ++l) {
                    auto local_info_pair = patch.localMatrix_idx(l);
                    impedance_matrix[i][l] = local_info_pair.first;
                    impedance_idx[i][l] = local_info_pair.second;
                }
            }
        }        
    }

    void smoothing(int l, Vec_t& u, Vec_t& rhs) {
        if(mw[l] * k_ < kh_threshold_) {
            block_GS(Op[l], rhs, u, dofs_perNode[l], 3);
        } else {
            for(int i = 0; i < u.size() / dofs_perNode[l]; ++i) {
                Mat_t Al = impedance_matrix[l][i];
                int local_idx = impedance_idx[l][i];
                Vec_t local_residual = Vec_t::Zero(Al.rows());
                local_residual.segment(local_idx*dofs_perNode[l], dofs_perNode[l]) = rhs.segment(i*dofs_perNode[l], dofs_perNode[l]);
                Vec_t el = Al.colPivHouseholderQr().solve(local_residual);
                u.segment(i*dofs_perNode[l], dofs_perNode[l]) += el.segment(local_idx*dofs_perNode[l], dofs_perNode[l]);
            }
        }
    }

public:
    double kh_threshold_;
    double k_;
    std::vector<double> mw;  // mesh_width
    std::vector<int> dofs_perNode;
    std::vector<SpMat_t> Op;
    std::vector<SpMat_t> I;   // prolongation operator
    std::unordered_map<int, std::vector<Mat_t>> impedance_matrix;
    std::unordered_map<int, std::vector<int>> impedance_idx;
};