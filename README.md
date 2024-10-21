# Iterative LQR

An implementation of the basic iLQR algorithm focused on code readability (not performance). It includes both the "standard" control-cost Hessian regularization proposed in Jacobson1970[^1] and the value function Hessian regularization from Tassa2012[^2]. The line-search implementation also follows Tassa2012[^2] and the regularization schedule is based on values of the line-search parameter[^3]. Currently, no form parallelization is implemented, but it will likely be added at some point in the future.

[^1]: Jacobson, David H. and Mayne, David Q. - *Differential Dynamic Programming.* - 1970.
[^2]: Tassa, Yuval and Erez, Tom and Todorov, Emanuel - *Synthesis and stabilization of complex behaviors through online trajectory optimization.* - 2012.
[^3]: The idea is not novel, but I do not recall the paper where it is introduced.

