# ADMM constraints
mutable struct Constraint{T}
    parameter::T                         # penalty parameter   
    projection::Union{Function,Nothing}  # projection function

    function Constraint{T}() where {T}
        return new{T}(one(T), nothing)
    end
end

struct Constraints{T}
    terminal_state::Constraint{T}
    zT::Vector{T}
    αT::Vector{T}

    input::Constraint{T}
    w::Vector{Vector{T}}
    β::Vector{Vector{T}}

    function Constraints{T}(nx, nu, N) where {T}
        terminal_state = Constraint{T}()
        zT = zeros(T, nx)
        αT = zeros(T, nx)

        input = Constraint{T}()
        w = [zeros(T, nu) for _ in 1:N]
        β = [zeros(T, nu) for _ in 1:N]

        return new(terminal_state, zT, αT, input, w, β)
    end
end

isactive(constraint::Constraint) = constraint.projection !== nothing

## projection setting functions
function set_terminal_state_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints.terminal_state, :projection, Π)
end

function set_input_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints.input, :projection, Π)
end

## penalty parameter setting functions
function set_terminal_state_constraint_parameter!(workset, ρ)
    @unpack parameter = workset.constraints.terminal_state
    @unpack αT = workset.constraints
    αT .*= parameter / ρ
    setproperty!(workset.constraints.terminal_state, :parameter, ρ)
    return nothing
end

function set_input_constraint_parameter!(workset, ρ)
    @unpack parameter = workset.constraints.input
    @unpack β = workset.constraints
    ThreadsX.map(β_k -> β_k ./= parameter / ρ, β)
    setproperty!(workset.constraints.input, :parameter, ρ)
    return nothing
end

## evaluation functions
function add_penalty_derivative!(gradient, hessian, constraint, primal, slack, dual)
    isactive(constraint) || return nothing
    gradient .+= constraint.parameter * (primal - slack + dual)
    hessian[diagind(hessian)] .+= constraint.parameter
    return nothing
end

function evaluate_penalty(constraint, residual, dual)
    isactive(constraint) || return nothing
    constraint.parameter / 2 * mapreduce(a -> a^2, +, residual + dual)
end

function update_slack_and_dual_variable!(constraint, primal, slack, dual)
    isactive(constraint) || return nothing
    slack .= constraint.projection(primal + dual)
    dual .+= primal - slack
end
