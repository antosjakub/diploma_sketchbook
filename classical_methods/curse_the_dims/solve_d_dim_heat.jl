using LinearAlgebra
using LinearOperators
using SparseArrays
using Serialization
include("main.jl")
include("pde_model.jl")

#type = ARGS[1]
#n = parse(Int, ARGS[2])
#d = parse(Int, ARGS[3])
#ft = parse(Int, ARGS[4])

type = "op" # op or sparse
n = 64
d = 4
ft = 32

serialize_result = false
solve_single_time_step = true
t_max = 1.0
nt = 100


FType = get_FloatType(ft)

t_max = FType(t_max)
h = FType( 1 / (n+1) )
tau = FType(t_max / (nt-1))
# need r <= 1 (implicit) or 1/2 (explicit)
alpha = FType(0.01)
r = FType(alpha * tau / h^2)

filename = "results/$type,n=$n,d=$d,ft=$ft"
N = n^d
println("================================")
println(filename)
println("N = $N")


using IterativeSolvers

## warm-up
n_warm_up = 10
N_warm_up = n_warm_up^d

if type == "sparse"
    L = get_laplace_sparse_matrix(n_warm_up, d, FType)
    A = sparse(1:N_warm_up, 1:N_warm_up, ones(FType,N_warm_up), N_warm_up, N_warm_up) - r * L;
elseif type == "op"
    a_operator! = create_operators(n_warm_up, d, r, FType)
    A = LinearOperator(FType, N_warm_up, N_warm_up, true, true, a_operator!)
else
    print("!!! ISSUE !!!")
end

Ut = Vector{FType}(undef, N_warm_up)
buffer = similar(Ut)
statevars = CGStateVariables(similar(Ut), similar(Ut), similar(Ut))

f_fun_2 = create_f_fun(d, FType)
u_analytic_fun_2 = create_u_fun(d, FType)
eval_on_grid!(n_warm_up, d, Ut, u_analytic_fun_2, FType(0.0))
eval_on_grid!(n_warm_up, d, buffer, f_fun_2, FType(0.0))
buffer .*= tau
Ut .+= buffer
cg!(
    buffer,
    A,
    Ut;
    statevars=statevars,
    verbose=true, maxiter=100
);
Ut .= buffer


using TimerOutputs
const to = TimerOutput()

if type == "sparse"
    @timeit to "L" L = get_laplace_sparse_matrix(n, d, FType)
    @timeit to "A" A = sparse(1:N, 1:N, ones(FType,N), N, N) - r * L;
elseif type == "op"
    @timeit to "create ops" a_operator! = create_operators(n, d, r, FType)
    @timeit to "A" A = LinearOperator(FType, N, N, true, true, a_operator!)
else
    print("!!! ISSUE !!!")
end

# intitialize
@timeit to "init Ut" Ut = Vector{FType}(undef, N)
@timeit to "init buffer" buffer = similar(Ut)
@timeit to "init cg vars" statevars = CGStateVariables(similar(Ut), similar(Ut), similar(Ut))
#
@timeit to "create f_fun" f_fun_2 = create_f_fun(d, FType)
@timeit to "create u_fun" u_analytic_fun_2 = create_u_fun(d, FType)
@timeit to "eval Ut" eval_on_grid!(n, d, Ut, u_analytic_fun_2, FType(0.0))
if serialize_result
    serialize("temp/U0.dat", Ut)
end
#tau = FType(t_max / (nt-1))
for ti in 2:1:nt
    # b = U^{t-1} + tau * F^t
    t = FType((ti-1)*tau)
    @timeit to "eval f" eval_on_grid!(n, d, buffer, f_fun_2, t)
    @timeit to "mult f" buffer .*= tau
    @timeit to "add to Ut" Ut .+= buffer
    @timeit to "solve CG" cg!(
        buffer,
        A,
        Ut;
        statevars=statevars,
        verbose=true, maxiter=100
    );
    @timeit to "post cg" Ut .= buffer
    if solve_single_time_step
        break
    end
end
if serialize_result
    if solve_single_time_step
        serialize("temp/U1.dat", Ut)
    else
        serialize("temp/UT.dat", Ut)
    end
end



using JSON
metadata = Dict("type" => type, "n" => n, "d" => d, "ft" => ft, "nt" => nt, "t_max" => t_max)
open("temp/metadata.json", "w") do file
    JSON.print(file, metadata, 4) # 4 = indent spaces
end

open(filename * ".txt", "w") do file
    show(file, to)
end

result = TimerOutputs.todict(to)
open(filename * ".json", "w") do file
    JSON.print(file, result, 4) # 4 = indent spaces
end
