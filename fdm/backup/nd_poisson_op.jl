using LinearAlgebra
using LinearOperators
using SparseArrays
include("main.jl")

n = 64
d = 2
N = n^d

#beta = 4.0 * pi
#gamma = 2.0
#alpha = gamma / beta^2
#
#function u_analytic_fun(t, x)
#    prod(sin.(beta*x)) * exp(- gamma * t)
#end
#

alpha = 0.01
k = 4.74
k2 = k^2
a = 10
beta = 1.0

function X(x)
    prod(sin.(pi*k*x))
end

function u_analytic_fun(x,t)
    X(x) * cos(a*t) * exp(-beta*t)
end

function f_fun(x,t)
    - ( a*sin(a*t) + (beta - alpha*pi^2*k2) ) * cos(a*t) * exp(-beta*t) * X(x)
end

h = 1 / (n+1)

t_max = 0.1
n_iters = 10
tau = t_max / (n_iters-1)
h, tau

# need r <= 1 (implicit) or 1/2 (explicit)
r = alpha * tau / h^2

grid_points_as_1d_vect = get_grid_points_as_1d_vect(n,d);

U_0 = u_analytic_fun.(grid_points_as_1d_vect, 0);

L = get_laplace_sparse_matrix(n,d);

r = alpha * tau / h^2

A_expl = sparse(I, N, N) + r * L;

U_evol_expl = zeros(n_iters, N)
U_evol_expl[1, :] = U_0
for ti in 2:1:n_iters
    t = (ti-1)*tau
    U_evol_expl[ti, :] = A_expl * U_evol_expl[ti-1, :] +
                         tau * f_fun.(grid_points_as_1d_vect, t-tau) +
                         r * compute_BC_corr(u_analytic_fun, t-tau, n, d);
end


using IterativeSolvers

A_impl = sparse(I, N, N) - r * L;

U_evol_impl = zeros(n_iters, N)
U_evol_impl[1, :] = U_0
for ti in 2:1:n_iters
    t = (ti-1)*tau
    U_evol_impl[ti, :] = cg(
        A_impl,
        U_evol_impl[ti-1, :] +
        tau * f_fun.(grid_points_as_1d_vect, t) +
        r * compute_BC_corr(u_analytic_fun, t-tau, n, d);
        verbose=true
    );
end


function a_op!(v_new, v)
    v_new .= v - r * laplace_operator(v)
end

A_op = LinearOperator(Float64, N, N, true, true, a_op!)

U_evol_op = zeros(n_iters, N)
U_evol_op[1, :] = U_0
for ti in 2:1:n_iters
    t = (ti-1)*tau
    U_evol_op[ti, :] = cg(
        A_op,
        U_evol_op[ti-1, :] +
        tau * f_fun.(grid_points_as_1d_vect, t) +
        r * compute_BC_corr(u_analytic_fun, t-tau, n, d);
        verbose=true
    );
end