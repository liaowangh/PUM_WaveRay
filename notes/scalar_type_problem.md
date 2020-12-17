There are four places concerning the SCALAR type.

```c++
auto fe_space = std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<SCALAR>>(mesh);
lf::uscalfe::ReactionDiffusionElementMatrixProvider<SCALAR ...>  elmat_builder(...);
lf::uscalfe::MassEdgeMatrixProvider<SCALAR ...> edge_mat_builder(...);
lf::uscalfe::ScalarLoadEdgeVectorProvider<SCALAR ...> edgeVec_builder(...);
```

When I set all the `SCALAR` to `double`, I got the error message:

` error: invalid operands to binary expression ('double' and 'const std::complex<double>')
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(DstScalar& a, const SrcScalar& b) const { a += b; }`

When assemble for the right hand size vector,
$$
\int_{\Gamma_R}gvdS
$$
 function `g` is complex-valued, and in `Eval()` function of `ScalarLoadEdgeVectorProvider` (line 858 of `loc_comp_ellvbp.h`)

```c++
template <class SCALAR, class FUNCTOR, class EDGESELECTOR>
 typename ScalarLoadEdgeVectorProvider<SCALAR, FUNCTOR, EDGESELECTOR>::ElemVec
 ScalarLoadEdgeVectorProvider<SCALAR, FUNCTOR, EDGESELECTOR>::Eval(
     const lf::mesh::Entity &edge) {
   /*
    ......
   */
   // Element vector
   ElemVec vec(pfe_.NumRefShapeFunctions());
   vec.setZero();
  
   auto g_vals = g_(edge, pfe_.Qr().Points());
  
   // Loop over quadrature points
   for (base::size_type k = 0; k < pfe_.Qr().NumPoints(); ++k) {
     // Add contribution of quadrature point to local vector
     const auto w = (pfe_.Qr().Weights()[k] * determinants[k]) * g_vals[k];
     vec += pfe_.PrecompReferenceShapeFunctions().col(k) * w;
   }
   return vec;
 }
```

the quadrature weight `w` is complex-valued because `g_vals[k]` is complex-valued.

in the update of `vec`

`vec += pfe_.PrecompReferenceShapeFunctions().col(k) * w;`

the scalar in left hand side if `double` while `std::complex<double>` in the right, so error occurs.

And if I set the SCALAR in`ScalarLoadEdgeVectorProvider` to complex

```c++
template <class SCALAR, class FUNCTOR, class EDGESELECTOR = base::PredicateTrue>
class ScalarLoadEdgeVectorProvider {
public:
using ElemVec = Eigen::Matrix<SCALAR, Eigen::Dynamic, 1>;
ScalarLoadEdgeVectorProvider(
std::shared_ptr<const UniformScalarFESpace<SCALAR>> fe_space, ...)
};
```

the constructor requires the `fe_space` to have the same scalar type, so if I pass the `fe_space` with type

`lf::uscalfe::FeSpaceLagrangeO1<double>`, I will receive the following message:

```
no known conversion for argument 1 from ‘shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>>’ to ‘shared_ptr<const lf::uscalfe::UniformScalarFESpace<std::complex<double> >>’
```

and the error message

```
/home/liaowang/.hunter/_Base/d45d77d/9d6aa46/5282022/Install/include/lf/uscalfe/precomputed_scalar_reference_finite_element.h:102:33: error: invalid covariant return type for ‘Eigen::MatrixXd lf::uscalfe::PrecomputedScalarReferenceFiniteElement<SCALAR>::EvalReferenceShapeFunctions(const MatrixXd&) const [with SCALAR = std::complex<double>; Eigen::MatrixXd = Eigen::Matrix<double, -1, -1>]’
  102 |   [[nodiscard]] Eigen::MatrixXd EvalReferenceShapeFunctions(
      |                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~
```