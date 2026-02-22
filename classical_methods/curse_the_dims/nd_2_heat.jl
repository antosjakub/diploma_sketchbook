using LinearAlgebra
using LinearOperators
using SparseArrays
include("main_2.jl")
include("pde_model.jl")

type = "sparse" # op or sparse
n = 64
d = 3
ft = 32
if ft == 16
    FType = Float16
elseif ft == 32
    FType = Float32
elseif ft == 64
    FType = Float64
else
    print("!!! ISSUE !!!")
end

filename = "results/$type,n=$n,d=$d,ft=$ft"
println(filename)
N = n^d
println("N = $N")


alpha = FType(0.01)

h = FType( 1 / (n+1) )

t_max = FType(0.1)
n_iters = 10
tau = FType(t_max / (n_iters-1))

# need r <= 1 (implicit) or 1/2 (explicit)
r = FType(alpha * tau / h^2)

using IterativeSolvers

## warm-up
n_warm_up = 10
N_warm_up = n_warm_up^d

L = get_laplace_sparse_matrix(n_warm_up, d, FType)
A = sparse(1:N_warm_up, 1:N_warm_up, ones(FType,N_warm_up), N_warm_up, N_warm_up) - r * L;

x = get_grid_points_as_1d_vect(n_warm_up, d, FType);
Ut = Vector{FType}(undef, N_warm_up)
buffer = similar(Ut)
Ut .= u_analytic_fun!(Ut, x, FType(0.0))
buffer .= f_fun!(buffer, x, FType(0.0))
buffer .*= tau
Ut .+= buffer
Ut .= cg(
    A,
    Ut;
    verbose=true, maxiter=100
);

# init for Ut - vector
# init for buffer - vector
# calc f - store to buffer
# Ut += tau * f
# 
# 

using TimerOutputs
const to = TimerOutput()

#function eval_on_grid!(vect, fun, t)
#end

@timeit to "L" L = get_laplace_sparse_matrix(n, d, FType)
@timeit to "A" A = sparse(1:N, 1:N, ones(FType,N), N, N) - r * L;

@timeit to "grid" x = get_grid_points_as_1d_vect(n, d, FType);
@timeit to "init Ut" Ut = Vector{FType}(undef, N)
@timeit to "init buffer" buffer = similar(Ut)
@timeit to "eval Ut" Ut .= u_analytic_fun!(Ut, x, FType(0.0))
@timeit to "eval f" buffer .= f_fun!(buffer, x, FType(0.0))
@timeit to "mult f" buffer .*= tau
@timeit to "add to Ut" Ut .+= buffer
@timeit to "solve CG" Ut .= cg(
    A,
    Ut;
    verbose=true, maxiter=100
);

#@timeit to "L" L = get_laplace_sparse_matrix(n,d,T);
#@timeit to "A" A = sparse(I, N, N) - r * L;
#
#@timeit to "grid" x = get_grid_points_as_1d_vect(n,d);
#@timeit to "Ut" Ut = u_analytic_fun(x, 0.0);
#@timeit to "u += f" Ut .+= tau * f_fun(x, 0.0);
#@timeit to "solve CG" Ut .= cg(
#    A,
#    Ut;
#    verbose=true
#);

#Ut = u_analytic_fun(x, 0.0);
#for ti in 2:1:n_iters
#    t = (ti-1)*tau
#    Ut .+= tau * f_fun(x, t);
#
#    Ut .= cg(
#        A,
#        Ut;
#        verbose=true
#    );
#end

#report_name = 

open(filename * ".txt", "w") do file
    show(file, to)
end


using JSON
result = TimerOutputs.todict(to)
open(filename * ".json", "w") do file
    JSON.print(file, result, 4) # 4 = indent spaces
end
