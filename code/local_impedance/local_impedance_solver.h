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
        int n = adjacent_vertex_.size(); // number of basis functions in vertex patch
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
        bool hole,lf::mesh::utils::CodimMeshDataSet<bool> inner_bdy, int degree=30): 
        N_wave(nr_waves), quad_degree(degree),
        vertex_patch_info(wave_number, mesh, hole, inner_bdy) {}

    std::pair<int, Eigen::SparseMatrix<double>> patch_idx_map(int l) override {
        int center_vertex_local_idx; // 
        int N_nodal = mesh_->NumEntities(2); // number of nodal global basis functions
        int n = adjacent_vertex_.size(); // number of nodal basis functions in vertex patch
        Eigen::SparseMatrix<double> Q(n * (N_wave + 1), N_nodal * (N_wave + 1));

        std::vector<int>& adj_vertex = adjacent_vertex_[l];
        std::vector<Eigen::Triplet<double>> triplets;
        for(int i = 0; i < adj_vertex.size(); ++i) {
            if(adj_vertex[i] == l) {
                center_vertex_local_idx = i;
            }
            for(int t = 0; t <= N_wave; ++t) {
                triplets.push_back(Eigen::Triplet<double>(i*(N_wave + 1)+t, adj_vertex[i]*(N_wave + 1)+t, 1.0));
            }
        }
        Q.setFromTriplets(triplets.begin(), triplets.end());
        return {center_vertex_local_idx, Q};
    }

    std::pair<Mat_t, int> localMatrix_idx(int l) override {
        auto patch_bdy = patch_boundary_selector(l);
        auto patch_element = patch_element_selector(l);
        auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_);
        lf::assemble::UniformFEDofHandler dofh(mesh_, {{lf::base::RefEl::kPoint(), N_wave}});
        int N_dofs(dofh.NumDofs());
        lf::assemble::COOMatrix<Scalar> A(N_dofs, N_dofs);

        // assembel for <grad u, grad v> - k^2 uv over vertex patch space l
        patch_ExtendPUM_ElementMatrix elmat_builder(N_wave-1, k_, 1.0, -k_ * k_, patch_element);
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