
using LinearAlgebra
using LinearOperators
using SparseArrays


function get_1d_laplace_sparse_matrix(n::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    off = ones(T, n-1)
    diag = -2 * ones(T, n)
    spdiagm(-1 => off, 0 => diag, 1 => off)
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

function get_laplace_sparse_matrix(n::Int, d::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    L = get_1d_laplace_sparse_matrix(n, T)
    if d == 1
        L_full = L
    elseif d == 2
        E = sparse(1:n, 1:n, ones(T,n), n, n)
        L_full = kron(E, L) + kron(L, E)
    elseif d >= 3
        # step 1
        size_ = n^(d-1)
        E = sparse(1:size_, 1:size_, ones(T,size_), size_, size_)
        L_full = kron(E, L)
        # step 2:d-1
        for k in 2:(d-1)
            n_Es_lhs = d-k
            n_Es_rhs = k-1
            # lhs
            size_lhs = n^n_Es_lhs
            E_lhs = sparse(1:size_lhs, 1:size_lhs, ones(T,size_lhs), size_lhs, size_lhs)
            kron_lhs = kron(E_lhs, L)
            # rhs
            size_rhs = n^n_Es_rhs
            E_rhs = sparse(1:size_rhs, 1:size_rhs, ones(T,size_rhs), size_rhs, size_rhs)
            L_full += kron(kron_lhs, E_rhs)
        end
        # step d
        size_ = n^(d-1)
        E = sparse(1:size_, 1:size_, ones(T,size_), size_, size_)
        L_full += kron(L, E)
    end
    return L_full
end


function create_operators(n::Int, d::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    """
    v -> Lv
    apply the L operator on vector v
    modifies both v and buffer
    """
    function laplace_operator!(v_new::Vector{T}, v::Vector{T})
        # create a different view - does not copy the underlaying data
        U  = reshape(v,     ntuple(_ -> n, d))
        LU = reshape(v_new, ntuple(_ -> n, d))

        fill!(LU, zero(eltype(v)))
        for dim in 1:d
            index_rs = ntuple(i -> i == dim ? (2:n)     : Colon(), d)
            index_ls = ntuple(i -> i == dim ? (1:(n-1)) : Colon(), d)
            LU[index_rs...] .+= U[index_ls...]
            LU[index_ls...] .+= U[index_rs...]
        end
        LU .-= 2d .* U

        v_new
    end

    # v_new = (I - r*L) v
    function a_operator!(v_new::Vector{T}, v::Vector{T})
        # v = input
        # v_new = output
        # ~ v_new = v - r * laplace_operator(v)
        laplace_operator!(v_new, v)
        v_new .*= - r
        v_new .+= v
    end

    return a_operator!
end
# n,d = ...
# a_operator! = create_operators(n,d)
#A = LinearOperator(Float64, N, N, true, true, a_operator!)




"""
returns Matrix of size (n^d x d)
- allocates nothing else
"""
function get_grid_points_as_1d_vect(n::Int, d::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    N = n^d
    h = T(1/(n+1))
    coords = Matrix{T}(undef, N, d)

    for k in 1:d
        inner = n^(k-1)
        period = n^k
        for i in 1:N
            coords[i, k] = (i-1) % period ÷ inner + 1
        end
    end

    return h .* coords
end


"""
    compute_BC_corr(u_D, t, n, h, d)

Vectorized construction of the boundary correction vector G^t_D ∈ ℝ^(n^d).
Iterates over the 2d boundary faces instead of all interior nodes.

Arguments:
  - u_D : function u_D(x, t) where x is a Vector of length d
  - t   : current time
  - n   : number of interior nodes per dimension
  - h   : grid spacing, h = 1/(n+1)
  - d   : number of spatial dimensions
"""
function compute_BC_corr(u_D, t, n, d)

    G = zeros(ntuple(_ -> n, d)) # shape (n, n, ..., n), will vec() at the end

    h = 1/(n+1)
    interior = (1:n) .* h # interior node coordinates along any axis

    for k in 1:d
        for (slice_k, x_k_val) in ((1, 0.0), (n, 1.0))

            # Build broadcast-compatible coordinate arrays for this face.
            # For dimension j ≠ k: shape is (1,...,n,...,1) with n in position j.
            # For dimension k: scalar x_k_val (the boundary coordinate).
            coords = ntuple(d) do j
                if j == k
                    x_k_val
                else
                    reshape(interior, ntuple(m -> m == j ? n : 1, d))
                end
            end

            # Evaluate u_D at every point on the face via broadcasting
            face_vals = broadcast((xs...) -> u_D(collect(xs), t), coords...)

            # Accumulate into the correct slice of G
            # (important: += because a corner node can get contributions from multiple faces)
            idx = ntuple(j -> j == k ? (slice_k:slice_k) : (1:n), d)
            G[idx...] .+= face_vals

        end
    end

    return vec(G)
end