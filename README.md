# 2D Unsteady Incompressible Navier-Stokes Solver (GLS-FEM)

A high-fidelity C++ implementation of a 2D Navier-Stokes solver developed for the **Mec645: Finite Element Method in Fluid Mechanics** course at Cairo University, Faculty of Engineering. This solver utilizes the **Galerkin/Least-Squares (GLS)** stabilization technique to simulate unsteady, incompressible rotational flows.

## Project Scope

This solver addresses the primary computational fluid dynamics (CFD) benchmark:

* **Lid-Driven Cavity (LDC):** A shear-driven flow within a square domain used to validate vortex dynamics and boundary layer accuracy.

## Key Features & Numerical Methods

* **Formulation:** Unsteady, 2D Incompressible Navier-Stokes equations.
* **Stabilization:** **Galerkin/Least-Squares (GLS)** scheme to handle advection-dominated regimes and prevent numerical pressure oscillations.
* **Time Integration:** Forward Euler scheme for temporal derivatives.
* **Low Dissipation:** Specifically tuned for accurate modeling of viscous effects in rotational flows.
* **Optimization:** The inverse of the pressure stiffness matrix () is pre-computed, enabling efficient field updates via matrix multiplication () rather than solving a linear system at every time step.

## Results & Validation (Lid-Driven Cavity)

### Performance Optimization

Pre-calculating the pressure stiffness matrix inverse significantly reduced execution time for fine meshes:

* **10x10 Mesh:** ~7.09 seconds
* **40x40 Mesh:** ~1222.74 seconds

### Grid Convergence Study

Data collected across various resolutions to ensure stability and accuracy:

| Mesh Size |  (Lower Boundary) |  (Lower Boundary) | Max Velocity (Y) |
| --- | --- | --- | --- |
| 10 x 10 | 0.000188 | -0.121376 | 0.282825 |
| 20 x 20 | 0.000108 | -0.000184 | 0.238713 |
| 30 x 30 | 0.000076 | 0.000132 | 0.232285 |
| 40 x 40 | 0.000058 | -0.030997 | 0.239099 |

## Tech Stack & Requirements

### Code Architecture

* **[Eigen Library](https://eigen.tuxfamily.org/):** Used for high-performance dense matrix operations (utilizing `FullPivLU` for matrix inversion).
* **OpenMP:** Integrated for multi-threaded execution and timing management.
* **VTK Output:** Custom `writeSolutionToVTK` function to export velocity and pressure fields for visualization in **ParaView**.

### Usage

* **Language:** C++
* **Inputs:** Mesh resolution, Reynolds Number (), Time step (), and Pressure Dissipation Parameter ().
* **Validation Case:** , ,  (Steady-state reached in ~0.56 seconds).

## Future Roadmap

To evolve this solver into a production-grade HPC tool, the following enhancements are planned:

* **Sparse Matrix Implementation:** Transitioning from dense matrices to **Sparse Matrix Storage (CSR/CSC)** using Eigen's Sparse modules to drastically reduce memory footprint and accelerate computation for large-scale meshes.
* **GPU Acceleration:** Enabling **CUDA/HIP kernels** to offload global matrix operations to the GPU, allowing for massive parallelism.
* **Complex Geometries:** Extending the non-Cartesian handling used in the cylinder validation to more complex boundary conditions.

---

### References

1. Pakdel P, Spiegelberg S H, McKinley G H. *Cavity flows of elastic liquids: two-dimensional flows*. Physics of Fluids 9, 11 (1997).
2. Hirsch, C. *Numerical computation of internal and external flows*. Butterworth-Heinemann, 2007.
3. Tezduyar, T. E. *Stabilized finite element formulations for incompressible flow computations*. Advances in applied mechanics 28 (1991).
