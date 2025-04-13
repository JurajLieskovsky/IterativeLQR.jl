# ADMM constraints
mutable struct Constraint{T}
    ρ::T                        # penalty parameter   
    Π::Union{Function,Nothing}  # projection function

    function Constraint{T}() where {T}
        return new{T}(one(T), nothing)
    end
end

struct Constraints{T}
    terminal_state::Constraint{T}
    z::Vector{T}
    α::Vector{T}

    input::Constraint{T}
    w::Vector{Vector{T}}
    β::Vector{Vector{T}}

    function Constraints{T}(nx, nu, N) where {T}
        terminal_state = Constraint{T}()
        z = zeros(T, nx)
        α = zeros(T, nx)

        input = Constraint{T}()
        w = [zeros(T, nu) for _ in 1:N]
        β = [zeros(T, nu) for _ in 1:N]

        return new(terminal_state, z, α, input, w, β)
    end
end

isactive(constraint::Constraint) = constraint.Π !== nothing

## workset functions
function set_projection_function!(workset, constraint::Symbol, projection::Function)
    setproperty!(getproperty(workset.constraints, constraint), :Π, projection)
end

function set_terminal_state_constraint_parameter!(workset, ρ_new)
    workset.constraints.α .*= workset.constraints.terminal_state.ρ / ρ_new
    workset.constraints.terminal_state.ρ = ρ_new
    return nothing
end

function set_input_constraint_parameter!(workset, ρ_new)
    ThreadsX.map(u -> u ./= workset.constraints.input.ρ / ρ_new, workset.constraints.β)
    workset.constraints.input.ρ = ρ_new
    return nothing
end

## evaluation functions
function add_penalty_derivative!(gradient, hessian, constraint, primal, slack, dual)
    isactive(constraint) || return nothing
    gradient .+= constraint.ρ * (primal - slack + dual)
    hessian[diagind(hessian)] .+= constraint.ρ
    return nothing
end

function evaluate_penalty(constraint, residual, dual)
    isactive(constraint) || return nothing
    constraint.ρ / 2 * mapreduce(a -> a^2, +, residual + dual)
end

function update_slack_and_dual_variable!(constraint, primal, slack, dual)
    isactive(constraint) || return nothing
    slack .= constraint.Π(primal + dual)
    dual .+= primal - slack
end
