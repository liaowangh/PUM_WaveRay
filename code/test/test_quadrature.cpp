#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>

#include <lf/base/base.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../utils/utils.h"

using namespace std::complex_literals;

int main(){
    std::string mesh_path = "../meshes/square.msh";
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    lf::io::GmshReader reader(std::move(mesh_factory), mesh_path);
   
    auto mesh = reader.mesh();

    double pi = std::acos(-1.);
    double k = 20.0;
    int N = 16;
    for(int i = 0; i < N; ++i) {
        Eigen::Vector2d d;
        d << std::cos(2*pi*i/N), std::sin(2*pi*i/N);
        auto f = [&d, &k](const Eigen::Vector2d& x)->Scalar {
            return std::exp(1i*k*d.dot(x));
        };
        std::cout << integrate(mesh, f, 20) << " " << L2_norm(mesh, f, 20) << std::endl;
    }

}