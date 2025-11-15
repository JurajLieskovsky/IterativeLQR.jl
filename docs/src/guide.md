# Getting started

To get started clone the repository into the `deps` folder of your project. Navigate 


## Examples

To run the examples we recommend cloning main repository
```
git clone git@github.com:JurajLieskovsky/IterativeLQR.jl.git IterativeLQR
```
 Navigating to the `IterativeLQR/examples` folder and starting Julia
```
cd IterativeLQR/examples
julia
```
In the REPL, activating and instantiating the environment
```
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
Running one of the examples
```
include("cartpole/swing_up.jl")
```
