using LinearAlgebra
using LinearOperators
using SparseArrays

function get_1d_laplace_op_matrix(n)
	off = ones(n-1)
	diag = ones(n)
	spdiagm(-1 => off, 0 => -2diag, 1 => off)
end

# L
# E L, L E
# E E L, E L E, L E E
# E E E L, E E L E, E L E E, L E E E
# (E E E L), (E E L) (E), (E L) (E E), (L) (E E E)
# E ... E L = L of size n^(k+1) if E is there k times
function get_laplace_op_matrix(n, d)
	L_full = get_1d_laplace_op_matrix(n^d)
	for k in 1:(d-1)
		L = get_1d_laplace_op_matrix(n^(d-k))
		size = n^k
		E = sparse(I, size, size)
		L_full += kron(L, E)
	end
	L_full
end


n = 100
d = 4
N = n^d

h = 1 / (n+1)


problem_statement = """Problem statement:
Delta u = f
BC: u = 0\
"""

"""\
u_analytic = sin(pi*x_1) * ... * sin(pi*x_d)\
"""
function u_analytic_fun(x)
    #prod(x)*prod(1 .- x)
    prod(sin.(pi*x))
end

"""\
f = -d*pi^2 * sin(pi*x_1) * ... * sin(pi*x_d) = -d*pi^2 * u\
"""
function f_fun(x)
    d = length(x)
    -d*pi^2 * prod(sin.(pi*x))
end

function get_grid_points_as_1d_vect(n, d)
    a = 0
    b = 1
    h = 1/(n+1)
    xs = [range(h, 1-h; length=n) for _ in 1:d]
    coords = collect(Iterators.product(xs...))
    [collect(x) for x in coords[1:end]]
end

using TimerOutputs

const to = TimerOutput()

println("1) Constructing a vector from the grid...")
@timeit to "1) construct grid vect" grid_points_as_1d_vect = get_grid_points_as_1d_vect(n,d);

println("2) Evaluating u_analytic_fun on the grid vector...")
@timeit to "2) eval u_analytic" U_analytic = u_analytic_fun.(grid_points_as_1d_vect);

println("3) Evaluating f_fun on the grid vector...")
@timeit to "3) eval f_fun" f = f_fun.(grid_points_as_1d_vect);
F = h^2 * f;

println("4) Constructing the laplace matrix...")
@timeit to "4) construct matrix" L = get_laplace_op_matrix(n,d);

println("5) Running CG...")
using IterativeSolvers
@timeit to "5) solve system" U_cg = cg(L, F)

#U_direct = L \ F;

show(to)
println()

open("timings.txt", "w") do file
	println(file, "n=$n")
	println(file, "d=$d")
	println(file, "N=$N")
	println(file)
	println(file, problem_statement)
	println(file)
	println(file, only((@doc u_analytic_fun).text) )
	println(file, only((@doc f_fun).text) )
	println(file)
    show(file, to)
end