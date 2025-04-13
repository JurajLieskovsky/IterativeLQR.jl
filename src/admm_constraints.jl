# Shared
function set_projection_function!(workset::Workset, constraint::Symbol, projection::Function)
    setproperty!(getproperty(workset, constraint), :projection, projection)
end

function set_penalty_parameter!(workset::Workset, constraint::Symbol, ρ::Number)
    set_penalty_parameter!(getproperty(workset, constraint), ρ)
end

# Single step constraint

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

function set_penalty_parameter!(constraint::SingleConstraint, ρ)
    constraint.dual .*= constraint.param / ρ
    constraint.param = ρ
    return nothing
end

function scale_penalty_parameter!(constraint::SingleConstraint, k)
    constraint.dual ./= k
    constraint.param *= k
    return nothing
end

function add_penalty_derivative!(gradient, hessian, constraint::SingleConstraint, primal)
    @unpack param, slack, dual = constraint
    gradient .+= param * (primal - slack + dual)
    hessian[diagind(hessian)] .+= param
    return nothing
end

function evaluate_penalty!(constraint::SingleConstraint, primal)
    @unpack param, slack, dual = constraint
    return param / 2 * mapreduce(a -> a^2, +, primal - slack + dual)
end

function update_slack_and_dual_variable!(constraint::SingleConstraint, primal)
    @unpack slack, dual, projection = constraint
    slack .= projection(primal + dual)
    dual .+= primal - slack
    return nothing
end

# Multiple step constraint

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

function set_penalty_parameter!(constraint::MultipleConstraint, ρ)
    ThreadsX.map(u -> u ./= constraint.param / ρ, constraint.dual)
    constraint.param = ρ
    return nothing
end

function scale_penalty_parameter!(constraint::MultipleConstraint, k)
    ThreadsX.map(u -> u ./= k, constraint.dual)
    constraint.param *= k
    return nothing
end

function add_penalty_derivative!(gradient, hessian, constraint::MultipleConstraint, primal)
    @unpack num, param, slack, dual = constraint
    @threads for k in 1:num
        gradient[k] .+= param * (primal[k] - slack[k] + dual[k])
        hessian[k][diagind(hessian[k])] .+= param
    end
    return nothing
end

function evaluate_penalty!(penalty, constraint::MultipleConstraint, primal)
    ThreadsX.map(
        (p, x, z, u) -> p = constraint.param / 2 * mapreduce(arg -> arg^2, +, x - z + u),
        penalty, primal, constraint.slack, constraint.dual
    )
    return nothing
end

function update_slack_and_dual_variable!(constraint::MultipleConstraint, primal)
    ThreadsX.map((x, z, u) -> z .= constraint.projection(x + u), primal, constraint.slack, constraint.dual)
    ThreadsX.map((x, z, u) -> u .+= x - z, primal, constraint.slack, constraint.dual)
    return nothing
end
