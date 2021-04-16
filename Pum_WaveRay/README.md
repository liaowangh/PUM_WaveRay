# Class HE_FEM

The base class, it stores the basic information about the Helmholtz problem and the elements for multigrid, part of the member variable:

```c++
std::vector<int> num_planewaves_;  // number of plan waves per mesh
size_type L_;  // number of refinement steps
double k_;  // wave number in the Helmholtz equation
    
std::shared_ptr<lf::io::GmshReader> reader_; // read the coarest mesh
std::shared_ptr<lf::refinement::MeshHierarchy> mesh_hierarchy_;
    // mesh_hierarchy_->getMesh(0) -- coarsest
    // mesh_hierarchy_->getMesh(L) -- finest
    
FHandle_t g_; // boundry data in Helmholtz equation
FHandle_t h_; // boundary data in Helmholtz equation
```

and the member functions do the following jobs:

- generate the prolongation operator between standard Lagrangian finite element spaces as well as the prolongation operator between standard Lagrangian finite element space and the plane wave PUM space.
- define a series of pure virtual function that is expeceted to be refined by the derived classes, including the functions to assemble the stiffness matrix, compute norms regarding different finite element spaces.



# Class PUM_WaveRay

this class is derived from two classes

```c++
class PUM_WaveRay: public HE_LagrangeO1, public HE_PUM
```

It contains the information for a series of finite element spaces, the space defined in finest mesh is the standard Lagrangian finite element space, and remaining are plane wave PUM method.