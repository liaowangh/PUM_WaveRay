# PUM-Based Wave-Ray Multigrid

This repository contains the codes for the master thesis project `PUM-Based Wave-Ray multigrid`,

this project aims at solving the 2-dimension Helmholtz equation.







## classes

```mermaid
classDiagram
	HE_FEM <|-- HE_PUM
	HE_FEM <|-- HE_LagrangeO1
	HE_FEM <|-- HE_ExtendPUM
	HE_LagrangeO1 <|-- PUM_WaveRay
	HE_PUM <|-- PUM_WaveRay
	HE_LagrangeO1 <|-- ExtendPUM_WaveRay
	HE_ExtendPUM <|-- ExtendPUM_WaveRay
	
	class HE_FEM{
		mesh_hierarchy_
		virtual build_equation()
		virtual prolongation()
		...
	}
	
	class HE_LagrangeO1{
	
	}
	
	class HE_PUM{
	
	}
	
	class HE_ExtendPUM{
	
	}
	
	class PUM_WaveRay{
	
	}
	
	class ExtendPUM_WaveRay{
	
	}
```









