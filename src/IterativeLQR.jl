module IterativeLQR

using Parameters
using LinearAlgebra
using Printf
using DataFrames, CSV
using Base.Threads
using GMW81

include("workset.jl")
include("trajectory_utils.jl")
include("regularization_functions.jl")
include("algorithm.jl")

end # module IterativeLQR
