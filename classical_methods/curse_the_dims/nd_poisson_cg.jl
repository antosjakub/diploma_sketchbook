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
function get_laplace_op_matrix(n, d)
	L = get_1d_laplace_op_matrix(n)
	if d == 1
		L_full = L
	elseif d == 2
		E = sparse(I, n, n)
		L_full = kron(E, L) + kron(L, E)
	elseif d >= 3
		# step 1
		size = n^(d-1)
		E = sparse(I, size, size)
		L_full = kron(E, L)
		# step 2:d-1
		for k in 2:(d-1)
			n_Es_lhs = d-k
			n_Es_rhs = k-1
			# lhs
			size = n^(n_Es_lhs)
			E = sparse(I, size, size)
			kron_lhs = kron(E, L)
			# rhs
			size = n^(n_Es_rhs)
			E = sparse(I, size, size)
			L_full += kron(kron_lhs, E)
		end
		# step d
		size = n^(d-1)
		E = sparse(I, size, size)
		L_full += kron(L, E)
	end
end


n = 64
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

println("1) Evaluating f_fun on the grid vector...")
@timeit to "1) eval f_fun" F = h^2 * f_fun.(get_grid_points_as_1d_vect(n,d));

println("2) Constructing the laplace matrix...")
@timeit to "2) construct matrix" L = get_laplace_op_matrix(n,d);

println("3) Running CG...")
using IterativeSolvers
@timeit to "3) solve system" U_cg = cg(L, F)

#U_direct = L \ F;
#grid_points_as_1d_vect = get_grid_points_as_1d_vect(n,d);
#U_analytic = u_analytic_fun.(grid_points_as_1d_vect);

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