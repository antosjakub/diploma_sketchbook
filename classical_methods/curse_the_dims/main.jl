
using LinearAlgebra
using LinearOperators
using SparseArrays


function get_1d_laplace_sparse_matrix(n)
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

# (E E E L), (E E L) (E), (E L) (E E), (L) (E E E)

# E E E x L
# (E E x L) x (E)
# (E x L) x (E E)
# L x E E E

function get_laplace_sparse_matrix(n, d)
	L = get_1d_laplace_sparse_matrix(n)
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


function laplace_operator(v)

    # from vector of value to values on a grid
    U = reshape(v, ntuple(i -> n, d)) 

    # apply the laplace stencil on every value on the grid
    LU = zeros(size(U));
    for dim in 1:d
        index_rs = ntuple(i -> i == dim ? (2:n) : Colon(), d)
        index_ls = ntuple(i -> i == dim ? (1:(n-1)) : Colon(), d)
        LU[index_rs...] .+= U[index_ls...]
        LU[index_ls...] .+= U[index_rs...]
    end
    LU .-= 2*d*U

    vec(LU)
end


function get_grid_points_as_1d_vect(n, d)
    a = 0
    b = 1
    h = 1/(n+1)
    xs = [range(h, 1-h; length=n) for _ in 1:d]
    coords = collect(Iterators.product(xs...))
    [collect(x) for x in coords[1:end]]
end