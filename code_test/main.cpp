#include <iostream>
#include <cmath>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

/*
 * generate sequence of nested triangular meshes with L levels
 */
std::shared_ptr<lf::refinement::MeshHierarchy>
generateMeshSequence(unsigned int L) {
    auto mesh = lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1. / 3.);
        // selector = 3: purely triangular mesh
    std::shared_ptr<lf::refinement::MeshHierarchy> meshes =
        lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh, L);
    return meshes;
}

lf::assemble::UniformFEDofHandler
generate_dof(std::shared_ptr<lf::refinement::MeshHierarchy> meshes,
             unsigned int L, unsigned int level) {
    auto mesh = meshes->getMesh(level);
    return lf::assemble::UniformFEDofHandler(mesh,
                       {{lf::base::RefEl::kPoint(), std::pow(2, L + 1 - level)}});
}

class ElementMatProvider {
public:
    ElementMatProvider(unsigned int L_, unsigned int l_, unsigned int k_):L(L_), l(l_), k(k_) {}
    virtual bool isActive(const lf::mesh::Entity& cell) { return true; }
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Eval(const lf::mesh::Entity& cell);
private:
    unsigned int L, l, k;
};

int main()
{
    unsigned int L = 4;
    auto meshes = generateMeshSequence(L);
    lf::assemble::UniformFEDofHandler dof_handler = generate_dof(meshes, L, 0);
    
    // return type of getMesh: std::shared_ptr<mesh::Mesh>
    auto mesh = meshes->getMesh(0);
    std::cout << "Mesh dimension: " << mesh->DimMesh() << std::endl
    << "Num of verties: " << mesh->NumEntities(2) << std::endl
    << "Num of segments: " << mesh->NumEntities(1) << std::endl
    << "Num of triangles: " << mesh->NumEntities(0) << std::endl;
    
    for(auto e : mesh->Entities(0)) {
        const lf::mesh::Entity &en(*e);
        Eigen::MatrixXd corners = lf::geometry::Corners(*(e->Geometry()));
        std::cout << "Corners:" << std::endl << corners << std::endl;
        std::cout << "Local dofs: " << dof_handler.NumLocalDofs(en) << std::endl;
        
        // Fetch unique index of current entity supplied by mesh object
        const lf::base::glb_idx_t e_idx = mesh->Index(en);
        // Number of local shape functions covering current entity
        const lf::assemble::size_type no_dofs(dof_handler.NumLocalDofs(en));
        // Obtain global indices of those shape functions ...
        nonstd::span<const lf::assemble::gdof_idx_t> dofarray{dof_handler.GlobalDofIndices(en)};
        // and print them
        std::cout << en << " " << e_idx << ": " << no_dofs << " dofs = [";
        for(int loc_dof_idx = 0; loc_dof_idx < no_dofs; ++loc_dof_idx) {
            std::cout << dofarray[loc_dof_idx] << ' ';
        }
        std::cout << " ]";
        break;
    }
    
    std::cout << "Total number of global basis functions " <<
        dof_handler.NumDofs() << std::endl;
//    std::cout << lf::base::RefEl::kTria().NumNodes() << std::endl;
    return 0;
}

