
using TimerOutputs
using LinearAlgebra
using LinearOperators
using SparseArrays


function get_1d_laplace_sparse_matrix(n::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    off = ones(T, n-1)
    diag = -2 * ones(T, n)
    spdiagm(-1 => off, 0 => diag, 1 => off)
end

function get_FloatType(ft::Int)
    if ft == 16
        return Float16
    elseif ft == 32
        return Float32
    elseif ft == 64
        return Float64
    else
        print("!!! ISSUE !!!")
        return null
    end
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

function create_operators(n::Int, d::Int, r::T, ::Type{T}=Float64) where {T<:AbstractFloat}
    #rs_idxs = [ntuple(i -> i == dim ? (2:n) : Colon(), Val(d)) for dim in 1:d]
    #ls_idxs = [ntuple(i -> i == dim ? (1:n-1) : Colon(), Val(d)) for dim in 1:d]
    #shape_ = ntuple(_ -> n, d)

    #function laplace_operator!(v_new::Vector{T}, v::Vector{T})
    #    U  = reshape(v, shape_)
    #    LU = reshape(v_new, shape_)

    #    fill!(LU, zero(T))
    #    for dim in 1:d
    #        @timeit to2 "1" LU[rs_idxs[dim]...] .+= U[ls_idxs[dim]...]
    #        @timeit to2 "2" LU[ls_idxs[dim]...] .+= U[rs_idxs[dim]...]
    #        #LU[rs_idxs[dim]...] .+= U[ls_idxs[dim]...]
    #        #LU[ls_idxs[dim]...] .+= U[rs_idxs[dim]...]
    #    end
    #    LU .-= 2 * d .* U
    #    show(to2)
    #    v_new
    #end

    #strides = (1, n, n^2)   # for d=3; in general: strides[k] = n^(k-1)

    # helper: convert Cartesian index (i₁,…,i_d) to linear index
    #@inline function linidx(I::NTuple{d,Int})
    #@inline function linidx(I)
    #    idx = 1
    #    s = 1
    #    @inbounds for k in 1:d
    #        idx += (I[k]-1)*s
    #        s *= n
    #    end
    #    return idx
    #end

    #to2 = TimerOutput()
    N = n^d
    #i = 0
    strides = ntuple(k -> n^(k-1), d)

    function laplace_operator!(v_new::Vector{T}, v::Vector{T})
        fill!(v_new, zero(T))
        
        Ik = 0
        @inbounds for idx in 1:N
            # decode linear index to multi-index
            #@timeit to2 "I $i" I = ntuple(k -> ((div(idx-1, strides[k]) % n) + 1), d)
            #I = ntuple(k -> ((div(idx-1, strides[k]) % n) + 1), d)

            u_center = v[idx]

            acc = zero(T)
            @inbounds for k in 1:d
                Ik = (div(idx-1, strides[k]) % n) + 1

                # + direction
                #if I[k] < n
                if Ik < n
                    acc += v[idx + strides[k]]
                end
                # - direction
                #if I[k] > 1
                if Ik > 1
                    acc += v[idx - strides[k]]
                end
            end

            v_new[idx] = acc - 2T(d) * u_center
        end
        #show(to2)
        #i += 1
        return v_new
    end

    function a_operator!(v_new::Vector{T}, v::Vector{T})
        # v = input
        # v_new = output
        # ~ v_new = v - r * laplace_operator(v)
        #@timeit to2 "a_op_1 $i" laplace_operator!(v_new, v)
        #@timeit to2 "a_op_2 $i" v_new .*= - r
        #@timeit to2 "a_op_3 $i" v_new .+= v
        laplace_operator!(v_new, v)
        v_new .*= - r
        v_new .+= v
        #show(to2)
        #i = i+1
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

function eval_on_grid!(n::Int, d::Int, v::Vector{T}, fun, t::T) where {T<:AbstractFloat}
    N = n^d
    h = T(1/(n+1))
    coords = Vector{T}(undef, d)

    @inbounds for i in 1:N
        for k in 1:d
            inner = n^(k-1)
            period = n^k
            coords[k] = (i-1) % period ÷ inner + 1
        end
        coords .*= h
        v[i] = fun(coords, t)
    end
    v
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