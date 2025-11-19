# Getting started

## Installation
The package can be added to your project using the command
```
] add https://github.com/JurajLieskovsky/IterativeLQR.jl.git
```
in your REPL. For usage, we recommend taking a look at one of the [examples](@repo/examples/cartpole/swing_up.jl].

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
