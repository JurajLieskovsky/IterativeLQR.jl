module IterativeLQR

using Parameters
using LinearAlgebra
using Printf
using DataFrames, CSV
using Infiltrator

include("workset.jl")
include("trajectory_utils.jl")
include("algorithm.jl")

end # module IterativeLQR
