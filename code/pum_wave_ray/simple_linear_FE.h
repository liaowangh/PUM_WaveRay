#pragma once
#include <functional>
#include <string>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>

using TriGeo_t = Eigen::Matrix<double, 2, 3>;
using FHandle_t = std::function<double<const Eigen::Vector2d&>;
using LocalMatrixHandle_t = std::function<Eigen::Matrix3d(const TriGeo_t&)>;
using LocalVectorHandle_t = std::function<Eigen::Vector3d(const TriGeo_t&, FHandle_t)>;
using Triplet_t = std::vector<Eigen::Triplet<double>>;

Eigen::Matrix3d ElementMatrix_Mass_LFE(const TriGeo_t& vertices);

/**
 * simple mesh data structure
 */

class TriMesh2D
{
public:
    // Constructor: reads mesh data from file
    TriMesh2D(const std::string&);
    virtual ~TriMesh2D() {}

    // Data members describing geometry and topolgy
    Eigen::Matrix<doyble, Eigen::Dynamic, 2> Coordinates;
    Eigen::Matrix<int, Eigen::Dynamic, 3> Elements;
};

Eigen::Matrix<double, 2, 3> gradbarycoordinates(const TriGeo_t& Vertices);

Eigen::Vector3d localLoadLFE(const TriGeo_t& Vertices, const FHandle_t& FHandle);

Eigen::Matrix3d ElementMatrix_Lapl_LFE(const TriGeo_t& Vertices);

Eigen::Matrix3d ElementMatrix_LaplMass_LFE(const TriGeo_t& Vertices);

Eigen::SparseMatrix<double> GalerkinAssembly(const TriaMesh2D& Mesh, const Local MatrixHandle_t& getElementMatrix);

Eigen::VectorXd assemble_LFE(const TriaMesh2D& Mesh, const LocalVectorHandle_t& getElementVector, const FHandle_t& FHandle);
