__precompile__()

module IRKGL_SIMD

using Reexport
@reexport using DiffEqBase

using LinearAlgebra
using Parameters
using OrdinaryDiffEq
using RecursiveArrayTools
using BenchmarkTools
using SIMD

export  IRKGL_simd, IRKGL_Seq, IRKAlgorithm
export  VecArray, floatType, IRKGLCoefficients
export  WPTests, launch_IRKGL_simd_tests
#export  launch_IRKGL_seq_tests, launch_IRKGL16_tests
export  launch_method_tests

include("IRKGL_Coefficients.jl")
include("IRKGL_Seq.jl")
include("IRKGL_SIMD_Solver.jl")
include("MyBenchmarksTools.jl")

end # module
