#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

int main()
{
    // Read *.msh_file
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const std::string mesh_path = "/home/liaowang/Documents/master-thesis/code_test/coarest_mesh.msh"; 
    lf::io::GmshReader reader(std::move(mesh_factory), mesh_path);
    // print all physical entities:
    std::cout << "Physical Entities in Gmsh File " << std::endl;
    std::cout
        << "---------------------------------------------------------------\n";
    for (lf::base::dim_t codim = 0; codim <= 2; ++codim) {
      for (auto& pair : reader.PhysicalEntities(codim)) {
        std::cout << "codim = " << static_cast<int>(codim) << ": " << pair.first
                   << " <=> " << pair.second << std::endl;
      }
    }
    std::cout << std::endl << std::endl;

    auto mesh = reader.mesh();
    std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy = 
        lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh, 3);
    auto finer_mesh = mesh_hierarchy->getMesh(1);

    auto outer_nr = reader.PhysicalEntityName2Nr("outer_boundary");
    auto inner_nr = reader.PhysicalEntityName2Nr("inner_boundary");

    std::vector<int> outer_idx, inner_idx;
    for(auto& e : finer_mesh->Entities(1)) {
        Eigen::MatrixXd corners = lf::geometry::Corners(*(e->Geometry()));
        std::cout << "Edge: " << finer_mesh->Index(*e) << std::endl << corners << std::endl;
        auto parent_e = mesh_hierarchy->ParentEntity(1, *e);
        if(reader.IsPhysicalEntity(*parent_e, outer_nr)) {
            outer_idx.push_back(finer_mesh->Index(*e));
        }
        if(reader.IsPhysicalEntity(*parent_e, inner_nr)) {
            inner_idx.push_back(finer_mesh->Index(*e));
        }
    }

    std::cout << "Index of Outer boundary:" << std::endl;
    for(int o : outer_idx)
        std::cout << o << " ";
    std::cout << std::endl;
    std::cout << "Index of Inner boundary:" << std::endl;
    for(int i : inner_idx)
        std::cout << i << " ";
    std::cout << std::endl;
	
}
