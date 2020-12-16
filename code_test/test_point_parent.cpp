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

    auto mesh = reader.mesh();
    std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy = 
        lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh, 3);
    auto finer_mesh = mesh_hierarchy->getMesh(1);

    auto outer_nr = reader.PhysicalEntityName2Nr("outer_boundary");
    auto inner_nr = reader.PhysicalEntityName2Nr("inner_boundary");

    for(unsigned int codim = 0; codim <= 2; ++codim) {
        for(const lf::mesh::Entity* entity: finer_mesh->Entities(codim)) {
            const lf::mesh::Entity* parent = mesh_hierarchy->ParentEntity(1, *entity);
            if(parent->RefEl() == lf::base::RefEl::kPoint()) {
                std::cout << "Parent of " << "Entity " << finer_mesh->Index(*entity) << " is also NODE" << std::endl;
            }

            // std::cout << "Entity " << finer_mesh->Index(*entity) << ": " << *entity << " ";
            // std::cout << "Parent " << *parent << std::endl;
        }
    }

    /*
    std::vector<int> outer_idx, inner_idx;
    for(const lf::mesh::Entity* point : finer_mesh->Entities(2)) {
        Eigen::MatrixXd corners = lf::geometry::Corners(*(point->Geometry()));
        std::cout << "Point: " << finer_mesh->Index(*point) << "[" << corners(0) << " " << corners(1) << "]"  << std::endl;
        auto parent_p = mesh_hierarchy->ParentEntity(1, *point);
        Eigen::MatrixXd p_corners = lf::geometry::Corners(*(parent_p->Geometry()));
        if(parent_p) {
            std::cout << "Parent: " << mesh->Index(*parent_p) << "[" << p_corners(0) << " "  << p_corners(1) << "]"  << std::endl;
        } else {
            std::cout << "No parent" << std::endl;
        }
        
        if(reader.IsPhysicalEntity(*parent_e, outer_nr)) {
            outer_idx.push_back(finer_mesh->Index(*e));
        }
        if(reader.IsPhysicalEntity(*parent_e, inner_nr)) {
            inner_idx.push_back(finer_mesh->Index(*e));
        }
        
    }
    */
}
