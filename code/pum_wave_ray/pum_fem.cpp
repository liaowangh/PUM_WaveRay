#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/io/io.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "EdgeMat.h"
#include "ElementMatrix.h"

lf::assemble::UniformFEDofHandler generate_dof(size_type level) {
    auto mesh = mesh_hierarchy->getMesh(level);
    size_type num = level == L ? 1 : std::pow(2, L + 1 - level);
    return lf::assemble::UniformFEDofHandler(mesh, {{lf::base::RefEl::kPoint(), num}});
}

