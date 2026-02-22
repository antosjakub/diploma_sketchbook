using LinearAlgebra
using LinearOperators
using SparseArrays
include("main_2.jl")
include("pde_model.jl")

n = 64
d = 4
N = n^d
println("N = $N")


alpha = 0.01

h = 1 / (n+1)

t_max = 0.1
n_iters = 10
tau = t_max / (n_iters-1)
h, tau

# need r <= 1 (implicit) or 1/2 (explicit)
r = alpha * tau / h^2

using IterativeSolvers

## warm-up
n_warm_up = 10
N_warm_up = n_warm_up^d
L = get_laplace_sparse_matrix(n_warm_up, d)
A_impl = sparse(I, N_warm_up, N_warm_up) - r * L;
x = get_grid_points_as_1d_vect(n_warm_up, d);
Ut = u_analytic_fun(x, 0.0)
Ut .+= tau * f_fun(x, 0.0)
Ut .= cg(
    A_impl,
    Ut;
    verbose=true
);


using TimerOutputs
const to = TimerOutput()

@timeit to "L" L = get_laplace_sparse_matrix(n,d);
@timeit to "A_impl" A_impl = sparse(I, N, N) - r * L;

@timeit to "grid" x = get_grid_points_as_1d_vect(n,d);
@timeit to "ut" Ut = u_analytic_fun(x, 0.0);
@timeit to "u += f" Ut .+= tau * f_fun(x, 0.0);
@timeit to "solve CG" Ut .= cg(
    A_impl,
    Ut;
    verbose=true
);

#Ut = u_analytic_fun(x, 0.0);
#for ti in 2:1:n_iters
#    t = (ti-1)*tau
#    Ut .+= tau * f_fun(x, t);
#
#    Ut .= cg(
#        A_impl,
#        Ut;
#        verbose=true
#    );
#end

open("timings_d=$d.txt", "w") do file
    show(file, to)
end
