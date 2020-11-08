#pragma once

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

class PUM_FEM {
public:
    using size_type = unsigned int;
    
    PUM_FEM(size_type L_, double k_, std::shared_ptr<lf::mesh::Mesh> mesh):
        L(L_), k(k_), coarsest_mesh(mesh),
    mesh_hierarchy(lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(coarsest_mesh, L)){}
    
    lf::assemble::UniformFEDofHandler generate_dof(size_type);
    
private:
    size_type L;  // number of refinement steps
    double k;  // wave number in the Helmholtz equation
    std::shared_ptr<lf::mesh::Mesh> coarsest_mesh; // pointer to the coarsest mesh, from which refinement will start
    std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy;
}
