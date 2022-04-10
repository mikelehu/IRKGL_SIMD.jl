__precompile__()

module IRKGL_SIMD

using Reexport
@reexport using DiffEqBase

using LinearAlgebra
using SIMD

const CompiledFloats = Union{Float32,Float64}

include("IRK16_seq.jl")
include("IRK16_SIMD.jl")

export IRK16, IRK16_SIMD
export CompiledFloats

end # module
