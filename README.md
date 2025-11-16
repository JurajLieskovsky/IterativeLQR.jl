# IterativeLQR.jl
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://jurajlieskovsky.github.io/IterativeLQR.jl/dev/)

An implementation of the iLQR algorithm focused mainly on code readability (not performance). Regularization of the problem, in the case of a noncovex running and/or final cost, is performed by minimally modifying their hessians. Differentiation as well as the afformentioned modification is performed in parallel assuming julia was started with multiple threads of execution.

