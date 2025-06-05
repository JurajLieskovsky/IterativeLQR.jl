# ADMM constraints
mutable struct Constraints{T}
    terminal_state_projection::Union{Function,Nothing}
    ρT::Vector{T}
    zT::Vector{T}
    αT::Vector{T}

    input_projection::Union{Function,Nothing}
    σ::Vector{Vector{T}}
    w::Vector{Vector{T}}
    β::Vector{Vector{T}}

    function Constraints{T}(nx, nu, N) where {T}
        terminal_state = nothing
        ρT = ones(T, nx)
        zT = zeros(T, nx)
        αT = zeros(T, nx)

        input = nothing
        σ = [ones(T, nu) for _ in 1:N]
        w = [zeros(T, nu) for _ in 1:N]
        β = [zeros(T, nu) for _ in 1:N]

        return new(terminal_state, ρT, zT, αT, input, σ, w, β)
    end
end

## projection setting functions
function set_terminal_state_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints, :terminal_state_projection, Π)
end

function set_input_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints, :input_projection, Π)
end

## penalty parameter setting functions
function set_constraint_parameter!(parameter, dual, new_parameter)
    dual .*= parameter ./ new_parameter
    parameter .= new_parameter
    return nothing
end

function set_terminal_state_constraint_parameter!(workset, ρ_new)
    @unpack αT, ρT = workset.constraints
    set_constraint_parameter!(ρT, αT, ρ_new)
    return nothing
end

function set_input_constraint_parameter!(workset, σ_new)
    @unpack σ, β = workset.constraints
    ThreadsX.map((σ_k, β_k) -> set_constraint_parameter!(σ_k, β_k, σ_new), σ, β)
    return nothing
end

## evaluation functions
function add_penalty_derivative!(gradient, hessian, parameter, primal, slack, dual)
    gradient .+= (primal - slack + dual) .* parameter
    hessian[diagind(hessian)] .+= parameter
    return nothing
end

function evaluate_penalty(parameter, primal, slack, dual)
    mapreduce((a, p) -> p / 2 * a^2, +, primal - slack + dual, parameter)
end

function update_slack_and_dual_variable!(projection, primal, slack, dual)
    slack .= projection(primal + dual)
    dual .+= primal - slack
    return nothing
end
