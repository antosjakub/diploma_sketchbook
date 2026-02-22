using LinearAlgebra


"""
sin(k1*x1)*...*sin(k1*x1) * cos(a*t)*exp(-beta*t)
Args:
    x - Matrix
    t - float
"""
function create_u_fun(d::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    alpha = T(0.01)
    k = 4*ones(T, d)
    a = T(10)
    beta = T(1.0)

    function u_fun(coords::Vector{T}, t::T) where {T<:AbstractFloat}
        out = T(1.0)
        for j in 1:d
            out *= sin(T(pi) * k[j] * coords[j])
        end
        out *= cos(a*t) * exp(-beta*t)
        out
    end

    return u_fun
end
function u_analytic_fun!(out::Vector{T}, x::Matrix{T}, t::T) where {T<:AbstractFloat}
    N, d = size(x)
    alpha = T(0.01)
    k = 4*ones(T, d)
    a = T(10)
    beta = T(1.0)

    out .= 1

    for j in 1:d
        kj = k[j]
        @inbounds for i in 1:N
            out[i] *= sin(T(pi) * kj * x[i,j])
        end
    end

    t_val = cos(a*t) * exp(-beta*t)
    @inbounds for i in 1:N
        out[i] *= t_val
    end

    return out
end


"""
X(x) = sin(k1*x1)*...*sin(k1*x1)
T(t) = - ( a*sin(a*t) + (beta - alpha*pi^2*k2)  * cos(a*t) ) * exp(-beta*t)
X(x) * T(t)
Args:
    x - Matrix
    t - float

f_fun(){Float64}
"""
function create_f_fun(d::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    alpha = T(0.01)
    k = 4*ones(T, d)
    k2 = sum(k.^2)
    a = T(10)
    beta = T(1.0)

    function f_fun(coords::Vector{T}, t::T) where {T<:AbstractFloat}
        out = T(1.0)
        for j in 1:d
            out *= sin(T(pi) * k[j] * coords[j])
        end
        out *= - ( a*sin(a*t) + (beta - alpha*T(pi)^2*k2)*cos(a*t) ) * exp(-beta*t)
        out
    end

    return f_fun
end
function f_fun!(out::Vector{T}, x::Matrix{T}, t::T) where {T<:AbstractFloat}
    N, d = size(x)
    alpha = T(0.01)
    k = 4*ones(T, d)
    k2 = sum(k.^2)
    a = T(10)
    beta = T(1.0)

    out .= 1

    for j in 1:d
        kj = k[j]
        @inbounds for i in 1:N
            out[i] *= sin(T(pi) * kj * x[i,j])
        end
    end

    t_val = - ( a*sin(a*t) + (beta - alpha*T(pi)^2*k2)*cos(a*t) ) * exp(-beta*t)
    @inbounds for i in 1:N
        out[i] *= t_val
    end

    return out
end