module IterativeLQR

using Parameters
using LinearAlgebra
using Printf
using DataFrames, CSV
using Infiltrator
using PositiveFactorizations
using Base.Threads
using DocStringExtensions

using GMW

include("workset.jl")
include("trajectory_utils.jl")
include("regularization_functions.jl")
include("algorithm.jl")

end # module IterativeLQR
