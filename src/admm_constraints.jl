# ADMM constraints

mutable struct SingleConstraint{T}
    param::T                             # penalty parameter   
    slack::Vector{T}                     # slack variables
    dual::Vector{T}                      # dual variables
    projection::Union{Function,Nothing}  # projection function

    function SingleConstraint{T}(n) where {T}
        slack = zeros(T, n)
        dual = zeros(T, n)
        return new{T}(one(T), slack, dual, nothing)
    end
end

mutable struct MultipleConstraint{T}
    num::Int64                           # number of constraints
    param::T                             # penalty parameter   
    slack::Vector{Vector{T}}             # slack variables
    dual::Vector{Vector{T}}              # dual variables
    projection::Union{Function,Nothing}  # projection function

    function MultipleConstraint{T}(n, m) where {T}
        slack = [zeros(T, n) for _ in 1:m]
        dual = [zeros(T, n) for _ in 1:m]
        return new{T}(m, one(T), slack, dual, nothing)
    end
end

## workset functions
function set_projection_function!(workset, constraint::Symbol, projection::Function)
    setproperty!(getproperty(workset, constraint), :projection, projection)
end

function set_penalty_parameter!(workset, constraint::Symbol, ρ::Number)
    set_penalty_parameter!(getproperty(workset, constraint), ρ)
end

## set penalty parameter
function set_penalty_parameter!(constraint::SingleConstraint, ρ)
    constraint.dual .*= constraint.param / ρ
    constraint.param = ρ
    return nothing
end

function set_penalty_parameter!(constraint::MultipleConstraint, ρ)
    ThreadsX.map(u -> u ./= constraint.param / ρ, constraint.dual)
    constraint.param = ρ
    return nothing
end

## add penalty derivatives
function add_penalty_derivative!(gradient, hessian, param, primal, slack, dual)
    gradient .+= param * (primal - slack + dual)
    hessian[diagind(hessian)] .+= param
end

function add_penalty_derivative!(gradient, hessian, constraint::SingleConstraint, primal)
    add_penalty_derivative!(gradient, hessian, constraint.param, primal, constraint.slack, constraint.dual)
    return nothing
end

function add_penalty_derivative!(gradient, hessian, constraint::MultipleConstraint, primal)
    ThreadsX.map(
        (g, H, x, z, u) -> add_penalty_derivative!(g, H, constraint.param, x, z, u),
        gradient, hessian, primal, constraint.slack, constraint.dual
    )
    return nothing
end

## evaluate penalty
evaluate_penalty(parameter, primal, slack, dual) = parameter / 2 * mapreduce(a -> a^2, +, primal - slack + dual)

function evaluate_penalty(constraint::SingleConstraint, primal)
    return evaluate_penalty(constraint.param, primal, constraint.slack, constraint.dual)
end

function evaluate_penalty!(penalty, constraint::MultipleConstraint, primal)
    ThreadsX.map(
        (p, x, z, u) -> p = constraint.param / 2 * mapreduce(arg -> arg^2, +, x - z + u),
        penalty, primal, constraint.slack, constraint.dual
    )
    return nothing
end

# update slack and dual variable
function update_slack_and_dual_variable!(projection, primal, slack, dual)
    slack .= projection(primal + dual)
    dual .+= primal - slack
end

function update_slack_and_dual_variable!(constraint::SingleConstraint, primal)
    update_slack_and_dual_variable!(constraint.projection, primal, constraint.slack, constraint.dual)
    return nothing
end

function update_slack_and_dual_variable!(constraint::MultipleConstraint, primal)
    ThreadsX.map(
        (x, z, u) -> update_slack_and_dual_variable!(constraint.projection, x, z, u),
        primal, constraint.slack, constraint.dual
    )
    return nothing
end
