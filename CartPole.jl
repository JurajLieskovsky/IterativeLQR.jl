module CartPole

using StaticArrays

const nx = 4    # number of states
const nu = 1    # number of inputs

struct Model
    g    # ms⁻² - gravity
    m_c  # kg - mass of the cart
    m_p  # kg - mass of the point-mass 
    l    # m - length of the pole
end

function f!(m, dx, x, u)
    M = @SMatrix [
        m.m_c+m.m_p m.m_p*m.l*cos(x[2])
        m.m_p*m.l*cos(x[2]) m.m_p*m.l^2
    ]

    τ = @SVector [
        m.m_p * m.l * sin(x[2]) * x[4]^2 + u[1],
        -m.g * m.m_p * m.l * sin(x[2])
    ]

    @views dx[1:2] .= x[3:4]
    @views dx[3:4] .= M \ τ

    return nothing
end

end

