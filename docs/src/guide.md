# Getting started

## Installation


To get started we recommend taking a look at one of the **examples**. (figure out how to link this)  

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
