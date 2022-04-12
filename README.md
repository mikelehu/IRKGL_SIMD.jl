# IRKGL_SIMD.jl

- SIMD-vectorized implementation of high order IRK integrators

We present a preliminary version of a SIMD-vectorized implementation of the sixteenth order 8-stage implicit Runge-Kutta integrator IRKGL16 implemented in the Julia package IRKGaussLegendre.jl. For numerical integrations of typical non-stiff problems performed in double precision, we show that a vectorized implementation of IRKGL16 that exploits the SIMD-based parallelism can clearly outperform high order explicit Runge-Kutta schemes available in the standard package DifferentialEquations.jl.


## Description

We present a preliminary version of a SIMD-vectorized implementation of the sixteenth order implicit Runge-Kutta integrator IRKGL16 implemented in the Julia package IRKGaussLegendre.jl.

The solver IRKGL16 is an implicit Runge-Kutta integrator of collocation type based on the Gauss-Legendre quadrature formula of 8 nodes. It is intended for high precision numerical integration of non-stiff systems of ordinary differential equations. The scheme has interesting properties that make it particularly useful for (but not limited to) long-term integrations of conservative problems. It is also useful for very accurate computations beyond the precision offered by the standard IEEE binary64 floating precision arithmetic.

For numerical integration of typical non-stiff problems performed in double precision floating point arithmetic, our sequential implementation of IRKGL16 is generally unable to outperform high order explicit Runge-Kutta schemes implemented in the standard package DifferentialEquations.jl. However, we show that a vectorized implementation of IRKGL16 that exploits the SIMD-based parallelism offered by modern processor can clearly outperform them. We demonstrate that by comparing our vectorized implementation of IRKGL16 with several high order explicit Runge-Kutta methods, such as Vern9, for different benchmark problems.

Our current implementation (https://github.com/mikelehu/IRKGL_SIMD.j) depends on the Julia package SIMD.jl to efficiently perform computations on vectors with eight Float64 numbers. The right-hand side of the system of ODEs to be integrated has to be implemented as a generic function defined in terms of the arithmetic operations and elementary functions implemented in the package SIMD.jl. The state variables must be collected in an array of Float64 or Float32 floating point numbers. The SIMD-based vectorization process is performed automatically under the hood.
