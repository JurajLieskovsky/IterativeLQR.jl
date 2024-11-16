module IterativeLQR

using Parameters
using LinearAlgebra
using Printf
using DataFrames, CSV

include("workset.jl")
include("trajectory_utils.jl")
include("algorithm.jl")

end # module IterativeLQR
