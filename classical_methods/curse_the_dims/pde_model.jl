using LinearAlgebra


"""
sin(k1*x1)*...*sin(k1*x1) * cos(a*t)*exp(-beta*t)
Args:
    x - Matrix
    t - float
"""
function u_analytic_fun(x::Matrix{Float64}, t::Float64)
    N, d = size(x)
    out = ones(N)
    alpha = 0.01
    k = 4*ones(d)
    a = 10
    beta = 1.0

    for j in 1:d
        kj = k[j]
        @inbounds for i in 1:N
            out[i] *= sin(pi * kj * x[i,j])
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
"""
function f_fun(x::Matrix{Float64}, t::Float64)
    N, d = size(x)
    out = ones(N)
    alpha = 0.01
    k = 4*ones(d)
    k2 = sum(k.^2)
    a = 10
    beta = 1.0

    for j in 1:d
        kj = k[j]
        @inbounds for i in 1:N
            out[i] *= sin(pi * kj * x[i,j])
        end
    end

    t_val = - ( a*sin(a*t) + (beta - alpha*pi^2*k2)*cos(a*t) ) * exp(-beta*t)
    @inbounds for i in 1:N
        out[i] *= t_val
    end

    return out
end