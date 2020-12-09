#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/SparseCore>

int main()
{
    // Read *.msh_file
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const std::string mesh_path = "/home/liaowang/Documents/master-thesis/code/pum_wave_ray/coarest_mesh.msh"; 

    auto u_sol = []()

    return 0;
    
}

