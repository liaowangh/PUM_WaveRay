#pragma once

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

#include "LoadVector.h"
#include "triangle_integration.h"

MassLoadVector::elem_vec_t Eval(const lf::mesh::Entity &edge) {
    LF_VERIFY_MSG(edge.RefEl() == lf::base::RefEl::kSegment(),
                  "Unsupported edge type " << edge.RefEl());
    // obtain endpoint coordinates of the triangle in a 2x2 matrix
    const auto endpoints = lf::geometry::Corners(*(edge.Geometry()));
    
    const double edge_length = (endpoints.col(1) - endpoints.col(0)).norm();
    
    size_type N = (1 << (L + 1 - 1));
}

