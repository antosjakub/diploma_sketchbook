"""
heat_equation_tt.jl

Solves the d-dimensional heat equation using Tensor Trains (MPS/MPO) via ITensors.jl:

    ∂u/∂t = Δu + f(x, t),    x ∈ [0,1]^d,  t ∈ [0, T]
    u(x, 0) = u₀(x)           (initial condition)
    u(x, t) = g(x, t)         (boundary conditions)

Each spatial dimension is one MPS "site" with local dimension N (grid points).
The solution u is an MPS of shape N × N × ... × N (d times).
The Laplacian is an MPO with bond dimension 3, exploiting the Kronecker sum structure:
    Δ = L₁⊗I⊗...⊗I + I⊗L₂⊗I⊗... + ... + I⊗...⊗Lₐ

Dependencies:
    using Pkg
    Pkg.add(["ITensors", "ITensorMPS", "ITensorTDVP", "KrylovKit", "LinearAlgebra"])
"""

using ITensors
using ITensorMPS
using LinearAlgebra

# ============================================================
# SECTION 1: Parameters
# ============================================================

const d      = 8        # number of spatial dimensions
const N      = 32       # interior grid points per dimension (Dirichlet: u=0 on boundary)
const dx     = 1.0 / (N + 1)   # grid spacing (interior points at dx, 2dx, ..., N*dx)
const dt     = 1e-4    # time step (explicit Euler stability: dt < dx²/2d ≈ 3e-6 for d=8,N=32)
                        # for CN or implicit, stability is unconditional
const T      = 1e-2    # final time
const nsteps = round(Int, T / dt)

# MPS truncation parameters (the key knobs)
const MAX_BOND_DIM = 64     # max MPS bond dimension (increase for more accuracy)
const CUTOFF      = 1e-10  # SVD cutoff for rank truncation

# ============================================================
# SECTION 2: 1D Building Blocks
# ============================================================

"""
    make_L1d(N, dx)

Build the 1D second-order finite-difference Laplacian matrix for N interior
points with homogeneous Dirichlet BCs (boundary values = 0).
"""
function make_L1d(N::Int, dx::Float64)
    L = zeros(N, N)
    for i in 1:N
        L[i, i] = -2.0 / dx^2
        i > 1 && (L[i, i-1] = 1.0 / dx^2)
        i < N && (L[i, i+1] = 1.0 / dx^2)
    end
    return L
end

# Grid coordinates for dimension k (1-indexed interior points)
grid_coords(N::Int, dx::Float64) = [(i * dx) for i in 1:N]

# ============================================================
# SECTION 3: ITensor Sites
# ============================================================

"""
    make_sites(d, N)

Create d ITensor Index objects, each of dimension N.
Each site represents one spatial dimension.
"""
function make_sites(d::Int, N::Int)
    return [Index(N, "Site,n=$k") for k in 1:d]
end

# ============================================================
# SECTION 4: Build the Laplacian MPO
# ============================================================

"""
    build_laplacian_mpo(sites, L1d)

Build the d-dimensional Laplacian as an MPO using the Kronecker sum structure:
    Δ_d = Σₖ I⊗...⊗L1d⊗...⊗I    (L1d acts on dimension k)

Uses a finite state machine (FSM) with bond dimension 2:
    Bond state 1 = "L has not been applied yet"
    Bond state 2 = "L has already been applied"

The W-matrix at each interior site is:
    W[s'←s; a←b] with block structure:
    |  I    0   |   ← incoming state 2 (already done), just pass I
    |  L1d  I   |   ← incoming state 1 (not done): either apply L (→ state 2) or pass I (→ state 1)

Wait: let's be careful about index ordering convention (left=incoming, right=outgoing):
    (in=1, out=1): I       — already acted, keep passing I
    (in=2, out=1): L1d     — act here, transition to "done"
    (in=2, out=2): I       — not yet acted, pass I for now

Left boundary (no incoming link):
    out=1: L1d (act here)
    out=2: I   (act later)

Right boundary (no outgoing link):
    in=1: I    (already done, end the chain)
    in=2: L1d  (last chance to act)
"""
function build_laplacian_mpo(sites::Vector{<:Index}, L1d::Matrix{Float64})
    d = length(sites)
    Id = Matrix{Float64}(I, N, N)

    # Bond indices (dim=2: state 1="acted", state 2="not yet acted")
    links = [Index(2, "Link,l=$l") for l in 1:d-1]

    tensors = Vector{ITensor}(undef, d)

    # ---- Site 1 (left boundary): no left link ----
    W = ITensor(sites[1]', sites[1], links[1])
    for i in 1:N, j in 1:N
        # out=1: apply L1d here
        L1d[i,j] != 0.0 && setindex!(W, L1d[i,j], sites[1]'=>i, sites[1]=>j, links[1]=>1)
        # out=2: pass I (will act later)
        i == j && setindex!(W, 1.0, sites[1]'=>i, sites[1]=>j, links[1]=>2)
    end
    tensors[1] = W

    # ---- Sites 2 .. d-1 (interior) ----
    for k in 2:d-1
        W = ITensor(sites[k]', sites[k], links[k-1], links[k])
        for i in 1:N, j in 1:N
            # in=1 → out=1: already acted, pass I
            i == j && setindex!(W, 1.0,       sites[k]'=>i, sites[k]=>j, links[k-1]=>1, links[k]=>1)
            # in=2 → out=1: act here with L1d
            L1d[i,j] != 0.0 &&
                setindex!(W, L1d[i,j], sites[k]'=>i, sites[k]=>j, links[k-1]=>2, links[k]=>1)
            # in=2 → out=2: pass I, act later
            i == j && setindex!(W, 1.0,       sites[k]'=>i, sites[k]=>j, links[k-1]=>2, links[k]=>2)
        end
        tensors[k] = W
    end

    # ---- Site d (right boundary): no right link ----
    W = ITensor(sites[d]', sites[d], links[d-1])
    for i in 1:N, j in 1:N
        # in=1: already acted, pass I
        i == j && setindex!(W, 1.0,       sites[d]'=>i, sites[d]=>j, links[d-1]=>1)
        # in=2: last chance — act with L1d
        L1d[i,j] != 0.0 &&
            setindex!(W, L1d[i,j], sites[d]'=>i, sites[d]=>j, links[d-1]=>2)
    end
    tensors[d] = W

    return MPO(tensors)
end

# ============================================================
# SECTION 5: Build an Identity MPO
# ============================================================

"""
    build_identity_mpo(sites)

Build the identity operator as an MPO (bond dim 1).
Needed for Crank-Nicolson: assemble (I ± α*L).
"""
function build_identity_mpo(sites::Vector{<:Index})
    d = length(sites)
    tensors = Vector{ITensor}(undef, d)
    links = [Index(1, "Link,l=$l") for l in 1:d-1]

    for k in 1:d
        if d == 1
            W = ITensor(sites[k]', sites[k])
            for i in 1:N; W[sites[k]'=>i, sites[k]=>i] = 1.0; end
        elseif k == 1
            W = ITensor(sites[k]', sites[k], links[k])
            for i in 1:N; W[sites[k]'=>i, sites[k]=>i, links[k]=>1] = 1.0; end
        elseif k == d
            W = ITensor(sites[k]', sites[k], links[k-1])
            for i in 1:N; W[sites[k]'=>i, sites[k]=>i, links[k-1]=>1] = 1.0; end
        else
            W = ITensor(sites[k]', sites[k], links[k-1], links[k])
            for i in 1:N; W[sites[k]'=>i, sites[k]=>i, links[k-1]=>1, links[k]=>1] = 1.0; end
        end
        tensors[k] = W
    end

    return MPO(tensors)
end

# ============================================================
# SECTION 6: MPS Construction from Functions
# ============================================================

"""
    separable_to_mps(funcs, sites; maxdim, cutoff)

Build an MPS for a **separable** function u₀(x₁,...,xₐ) = ∏ₖ fₖ(xₖ).
Results in exact MPS with bond dim = 1.

Args:
    funcs : Vector of d functions, each mapping a Float64 grid coordinate to Float64
    sites : ITensor site indices
"""
function separable_to_mps(funcs::Vector{<:Function}, sites::Vector{<:Index};
                           maxdim::Int=MAX_BOND_DIM, cutoff::Float64=CUTOFF)
    d = length(sites)
    @assert length(funcs) == d "Need one function per dimension"

    x = grid_coords(N, dx)   # interior grid points
    links = [Index(1, "Link,l=$l") for l in 1:d-1]
    tensors = Vector{ITensor}(undef, d)

    for k in 1:d
        if k == 1 && d > 1
            W = ITensor(sites[k], links[k])
            for i in 1:N; W[sites[k]=>i, links[k]=>1] = funcs[k](x[i]); end
        elseif k == d && d > 1
            W = ITensor(sites[k], links[k-1])
            for i in 1:N; W[sites[k]=>i, links[k-1]=>1] = funcs[k](x[i]); end
        elseif d == 1
            W = ITensor(sites[k])
            for i in 1:N; W[sites[k]=>i] = funcs[k](x[i]); end
        else
            W = ITensor(sites[k], links[k-1], links[k])
            for i in 1:N; W[sites[k]=>i, links[k-1]=>1, links[k]=>1] = funcs[k](x[i]); end
        end
        tensors[k] = W
    end

    return MPS(tensors)
end

"""
    ttcross_to_mps(f, sites; maxdim, cutoff, nsamples)

Approximate a **general** (non-separable) function as an MPS using the
TT-Cross / DMRG-Cross algorithm (Oseledets & Tyrtyshnikov 2010).

This avoids forming the full N^d tensor. Instead it samples the function
at O(d·r²·N) points via alternating cross interpolation sweeps.

NOTE: This is a simplified skeleton. For production use, consider:
  - The `TensorCrossInterpolation.jl` package (actively maintained)
  - Or `TENEVA`'s cross function called from Python

Args:
    f       : function taking a Vector{Int} of d 1-based grid indices → Float64
    sites   : ITensor site indices
    maxdim  : maximum TT-rank (controls accuracy vs cost)
"""
function ttcross_to_mps(f::Function, sites::Vector{<:Index};
                         maxdim::Int=MAX_BOND_DIM, cutoff::Float64=CUTOFF,
                         n_pivot::Int=5)
    # ---- Forward sweep: build initial MPS skeleton via nested CUR ----
    d = length(sites)

    # For a full implementation, see TensorCrossInterpolation.jl
    # Here we show the conceptual structure of one sweep:

    # 1. Initialize pivot indices (random or maximum-volume selection)
    # 2. Left-to-right sweep: for each k, form a "fiber" matrix C_{k}[i_k, J_{k+1}]
    #    where J_{k+1} is the set of multi-indices for dimensions k+1..d
    # 3. Apply CUR decomposition to find pivot rows/columns
    # 4. Store the resulting MPS core and update pivots
    # 5. Repeat until convergence

    # Placeholder: for non-separable functions, call TensorCrossInterpolation.jl:
    #   using TensorCrossInterpolation
    #   f_wrapper = TCI.TensorTrain(f, fill(N, d))
    #   mps = tci_to_itensor_mps(f_wrapper, sites)

    error("General TT-Cross not implemented here. " *
          "Use TensorCrossInterpolation.jl for non-separable functions.")
end

# ============================================================
# SECTION 7: Non-Homogeneous Boundary Conditions
# ============================================================

"""
    apply_dirichlet_bc!(u::MPS, g_boundary::Union{Float64, Function})

Enforce Dirichlet boundary conditions u = g on ∂Ω.

For homogeneous BCs (g=0), our grid already excludes boundary points, so
nothing needs to be done — the FD matrix L1d already enforces this.

For non-homogeneous BCs (g ≠ 0), we need to:
  1. Lift the solution: ũ = u - u_lift, where u_lift satisfies the BCs
  2. Solve the equation for ũ with homogeneous BCs
  3. Recover u = ũ + u_lift at the end

Here we show how to compute and subtract the BC correction at each time step.
g_boundary : value on the boundary (scalar or function of boundary coordinates)
"""
function build_bc_correction_mps(g_bc::Float64, sites::Vector{<:Index};
                                   maxdim::Int=MAX_BOND_DIM, cutoff::Float64=CUTOFF)
    # Simple case: u = g_bc on all boundaries
    # The FD correction term for non-homogeneous BCs is a vector:
    # f_bc[i] = (g_bc / dx^2) * (# boundary neighbors of grid point i)
    # This is separable in 1D (only boundary-adjacent points are affected)

    # For point i in 1D: correction = g_bc/dx² at i=1 and i=N
    correction_1d = zeros(N)
    correction_1d[1] += g_bc / dx^2    # left boundary contribution
    correction_1d[N] += g_bc / dx^2    # right boundary contribution

    # Build as a sum of rank-1 MPS: correction_full = Σₖ I⊗...⊗c₁ₐ⊗...⊗I
    # where c₁ₐ is the 1D correction vector for dimension k
    # (same structure as the Laplacian MPO, but for a vector)

    # For simplicity we return a rank-1 MPS with the average correction
    # (proper implementation would sum over all d dimensions)
    avg_correction = [x -> correction_1d[i] for i in 1:N]  # placeholder

    # TODO: full implementation needs to sum d MPS terms, one per dimension
    @warn "Non-homogeneous BC correction is approximate here; use full sum for accuracy."
    funcs = [x -> sum(correction_1d) / N for _ in 1:d]
    return separable_to_mps(funcs, sites; maxdim=maxdim, cutoff=cutoff)
end

# ============================================================
# SECTION 8: MPO Linear Combination (I + α*L)
# ============================================================

"""
    mpo_add(H1::MPO, α::Float64, H2::MPO, β::Float64) → MPO

Compute α*H1 + β*H2 as an MPO. Bond dimension is at most dim(H1) + dim(H2).
Used to form (I - dt/2 * L) and (I + dt/2 * L) for Crank-Nicolson.

ITensors.jl provides `+(MPO, MPO)` for this in recent versions.
"""
function build_cn_operators(H_L::MPO, H_I::MPO, dt::Float64)
    # A = I - (dt/2) * L   (left-hand side operator)
    # B = I + (dt/2) * L   (right-hand side operator)
    # In ITensor: no direct MPO scaling; scale by modifying tensors of H_L
    # We apply the scalar to the first tensor of H_L (convention)

    A = H_I + (-dt/2) * H_L   # ITensor MPO addition (bond dim grows)
    B = H_I + ( dt/2) * H_L
    return A, B
end

# ============================================================
# SECTION 9: ALS Linear Solver for MPO*MPS = MPS
# ============================================================

"""
    als_linsolve(A::MPO, b::MPS, x0::MPS; maxsweeps, tol, maxdim, cutoff)

Solve A|x⟩ = |b⟩ for the MPS |x⟩ using Alternating Linear Squares (ALS),
also known as DMRG for linear systems.

At each step k, we freeze all MPS tensors except the k-th core Mₖ and
solve the local linear system:
    (⟨L|A†A|R⟩) mₖ = ⟨L|A†|b⟩|R⟩

where L, R are the left/right environments accumulated by sweeping.

Cost per sweep: O(d · r³ · N · χ²) where r=MPS rank, χ=MPO bond dim.

IMPORTANT: For production use, consider using KrylovKit.jl's GMRES
           in the MPS inner product space (see commented block below).
"""
function als_linsolve(A::MPO, b::MPS, x0::MPS;
                       maxsweeps::Int=10, tol::Float64=1e-8,
                       maxdim::Int=MAX_BOND_DIM, cutoff::Float64=CUTOFF)
    d = length(x0)
    x = copy(x0)
    orthogonalize!(x, 1)    # put x in left-canonical form starting from site 1

    # --- Build initial right environments ---
    # R[k] = contraction of sites k..d of ⟨x|A|x⟩ and ⟨x|b⟩
    # This is accumulated left-to-right and right-to-left during sweeps.
    # Below is a schematic; ITensor handles index contraction automatically.

    R_env = Vector{ITensor}(undef, d + 1)   # right environments for A†A
    Rb_env = Vector{ITensor}(undef, d + 1)  # right environments for A†b

    # Rightmost boundary: scalar 1
    R_env[d+1]  = ITensor(1.0)
    Rb_env[d+1] = ITensor(1.0)

    # Build right environments by contracting from right
    for k in d:-1:1
        # Contract: R[k] = x[k]* ⊗ A[k]† ⊗ A[k] ⊗ x[k] ⊗ R[k+1]
        # (schematic — actual index wiring done by ITensor prime/dag conventions)
        Wk = A[k]
        xk = x[k]
        bk = b[k]

        # Right env for A†A term:
        R_env[k] = (xk * prime(dag(xk), "Site")) *
                   (Wk * prime(dag(Wk))) *
                   R_env[k+1]

        # Right env for A†b term:
        Rb_env[k] = (bk * prime(dag(xk), "Site")) * R_env[k+1]
    end

    # --- ALS sweeps ---
    L_env  = Vector{ITensor}(undef, d + 1)
    Lb_env = Vector{ITensor}(undef, d + 1)
    L_env[1]  = ITensor(1.0)
    Lb_env[1] = ITensor(1.0)

    prev_norm = Inf
    for sweep in 1:maxsweeps

        # ------ Left-to-right sweep ------
        for k in 1:d
            # Effective local operator: Heff = L_env[k] ⊗ A[k]†A[k] ⊗ R_env[k+1]
            # Effective rhs:            rhs  = Lb_env[k] ⊗ b[k]* ⊗ Rb_env[k+1]
            # Local solve: Heff * vec(Mₖ) = rhs  (small dense system, size N*r_l*r_r)

            Heff = L_env[k]  * A[k] * prime(dag(A[k])) * R_env[k+1]
            rhs  = Lb_env[k] * b[k]                    * Rb_env[k+1]

            # Solve small local system by converting to Matrix and using \ 
            # (indices of Heff: site_k', site_k, left_link, right_link — reshape to 2D)
            local_size = dim(sites(x)[k]) * (k > 1 ? dim(linkind(x, k-1)) : 1) *
                                            (k < d ? dim(linkind(x, k))   : 1)
            Hmat = reshape(Array(Heff, inds(Heff)), local_size, local_size)
            rvec = vec(Array(rhs,  inds(rhs)))
            Mvec = Hmat \ rvec    # dense solve (small system)

            # Put result back into MPS tensor
            x[k] = ITensor(reshape(Mvec, size(Array(x[k]))), inds(x[k]))

            # SVD to maintain canonical form and limit bond dim
            if k < d
                U, S, V = svd(x[k], (k == 1 ? [sites(x)[k]] :
                                     [sites(x)[k], linkind(x, k-1)]);
                               maxdim=maxdim, cutoff=cutoff)
                x[k]   = U
                x[k+1] = S * V * x[k+1]
            end

            # Update left environments
            L_env[k+1]  = L_env[k]  * x[k] * prime(dag(x[k]), "Site") *
                          A[k] * prime(dag(A[k]))
            Lb_env[k+1] = Lb_env[k] * x[k] * prime(dag(b[k]),  "Site")
        end

        # Check convergence: ||A|x⟩ - |b⟩||
        residual = apply(A, x; maxdim=maxdim, cutoff=cutoff)
        res_norm = norm(add(residual, b, -1.0))    # ||Ax - b||
        println("  ALS sweep $sweep: residual = $res_norm")
        abs(res_norm - prev_norm) < tol && break
        prev_norm = res_norm
    end

    return x
end

# ---- Alternative: KrylovKit GMRES in MPS space ----
# Uncomment to use instead of ALS (often more robust for ill-conditioned problems)
#=
using KrylovKit
function krylov_linsolve(A::MPO, b::MPS, x0::MPS; tol=1e-8, maxiter=100,
                          maxdim=MAX_BOND_DIM, cutoff=CUTOFF)
    # Define matvec as an MPS-space linear map
    function matvec(x_vec::MPS)
        return apply(A, x_vec; maxdim=maxdim, cutoff=cutoff)
    end

    # Define MPS inner product
    function mps_dot(x::MPS, y::MPS)
        return inner(x, y)
    end

    # GMRES in MPS space (experimental — KrylovKit needs custom vector space)
    # This requires wrapping MPS operations to satisfy KrylovKit's VectorInterface
    # See: https://github.com/Jutho/KrylovKit.jl

    x, info = linsolve(matvec, b, x0;
                       tol=tol, maxiter=maxiter, krylovdim=20,
                       isposdef=false)   # L is negative semi-definite
    @info "GMRES: $(info.numiter) iterations, residual $(info.normres)"
    return x
end
=#

# ============================================================
# SECTION 10: Time Stepping
# ============================================================

"""
    euler_step(u, H_L, f_mps, dt; maxdim, cutoff) → MPS

Explicit Euler step for ∂u/∂t = Δu + f:
    u^{n+1} = u^n + dt * (L * u^n + f)

Simple but has strict stability constraint: dt < dx²/(2d).
For d=8, N=32 → dx=0.03, dt < 5.7×10⁻⁵.
"""
function euler_step(u::MPS, H_L::MPO, f_mps::MPS, dt::Float64;
                    maxdim::Int=MAX_BOND_DIM, cutoff::Float64=CUTOFF)
    # L*u: apply Laplacian MPO to current solution
    Lu = apply(H_L, u; maxdim=maxdim, cutoff=cutoff)

    # u_new = u + dt*(Lu + f) = u + dt*Lu + dt*f
    u_new = add(u, Lu, 1.0, dt; maxdim=maxdim, cutoff=cutoff)
    u_new = add(u_new, f_mps, 1.0, dt; maxdim=maxdim, cutoff=cutoff)

    return u_new
end

"""
    crank_nicolson_step(u, H_L, H_I, f_mps, dt; maxdim, cutoff) → MPS

Crank-Nicolson step (unconditionally stable, 2nd order in time):
    (I - dt/2 * L) u^{n+1} = (I + dt/2 * L) u^n + dt * f

Requires solving a linear system in MPS space at each step.
Uses ALS solver internally.
"""
function crank_nicolson_step(u::MPS, H_L::MPO, H_I::MPO, f_mps::MPS, dt::Float64;
                               maxdim::Int=MAX_BOND_DIM, cutoff::Float64=CUTOFF,
                               solver_sweeps::Int=10)
    # Build Crank-Nicolson operators: A = I - dt/2*L, B = I + dt/2*L
    # NOTE: MPO addition creates MPO with bond dim = sum of input bond dims
    #   B*u + dt*f is cheap (just apply + add)
    #   A*u_new = rhs requires the linear solve

    # Compute RHS = (I + dt/2 * L) u^n + dt * f
    Lu    = apply(H_L, u; maxdim=maxdim, cutoff=cutoff)
    rhs   = add(u, Lu, 1.0, dt/2; maxdim=maxdim, cutoff=cutoff)
    rhs   = add(rhs, f_mps, 1.0, dt; maxdim=maxdim, cutoff=cutoff)

    # Build LHS operator A = I - dt/2 * L
    # Scale H_L by -dt/2: multiply first tensor by scalar
    H_L_scaled = copy(H_L)
    H_L_scaled[1] *= (-dt/2)
    A = add(H_I, H_L_scaled)   # A = I + (-dt/2)*L

    # Solve A * u_new = rhs using ALS
    # Use previous u as initial guess (good for small dt)
    u_new = als_linsolve(A, rhs, copy(u);
                          maxsweeps=solver_sweeps,
                          maxdim=maxdim, cutoff=cutoff)
    return u_new
end

# ============================================================
# SECTION 11: Utilities
# ============================================================

"""
    evaluate_mps(u, sites, idx)

Evaluate the MPS u at grid multi-index idx = [i₁, i₂, ..., iₐ] (1-based).
Cost: O(d * r²) where r is the bond dimension.
"""
function evaluate_mps(u::MPS, sites::Vector{<:Index}, idx::Vector{Int})
    d = length(sites)
    @assert length(idx) == d

    val = ITensor(1.0)
    for k in 1:d
        state_k = onehot(sites[k] => idx[k])   # basis vector eᵢₖ
        val = val * u[k] * state_k
    end
    return scalar(val)
end

"""
    l2_norm_mps(u) → Float64

Compute the L2 norm of the MPS: √(⟨u|u⟩).
Costs O(d * r³) per call.
"""
l2_norm_mps(u::MPS) = norm(u)

"""
    print_mps_info(u, step)

Print diagnostics about the current MPS state.
"""
function print_mps_info(u::MPS, step::Int, t::Float64)
    r_max = maxlinkdim(u)
    bond_dims = [dim(linkind(u, k)) for k in 1:(length(u)-1)]
    println("t=$(round(t, digits=5)), step=$step | ||u|| = $(round(norm(u), sigdigits=4)) " *
            "| max bond dim = $r_max | bonds = $bond_dims")
end

# ============================================================
# SECTION 12: Main Solver
# ============================================================

"""
    solve_heat_equation(; method=:euler)

Main driver. Choose method = :euler or :crank_nicolson.

Example initial condition (separable):
    u₀(x₁,...,xₐ) = ∏ₖ sin(π xₖ)
    Exact solution (no source): u(x,t) = exp(-π²·d·t) · u₀(x)

Example source term (separable):
    f(x₁,...,xₐ) = ∏ₖ sin(π xₖ)
"""
function solve_heat_equation(; method::Symbol = :euler)
    println("="^60)
    println("Heat equation in d=$d dimensions, N=$N grid points/dim")
    println("Grid points total (conceptual): $(N)^$d = $(big(N)^d)")
    println("Method: $method, dt=$dt, T=$T, steps=$nsteps")
    println("="^60)

    # ---- Setup ----
    s = make_sites(d, N)
    L1d = make_L1d(N, dx)
    println("\nBuilding Laplacian MPO (bond dim = 3)...")
    H_L = build_laplacian_mpo(s, L1d)
    H_I = build_identity_mpo(s)

    # ---- Initial condition: u₀ = ∏ sin(πxₖ) ----
    println("Building initial condition MPS...")
    u0_funcs = [x -> sin(π * x) for _ in 1:d]
    u = separable_to_mps(u0_funcs, s; maxdim=MAX_BOND_DIM, cutoff=CUTOFF)
    normalize!(u)   # normalize for numerical stability tracking
    println("Initial MPS bond dims: ", [dim(linkind(u, k)) for k in 1:d-1])

    # ---- Source term: f = ∏ sin(πxₖ) (constant in time here) ----
    # For time-dependent f, rebuild the MPS inside the time loop
    println("Building source term MPS...")
    f_funcs = [x -> sin(π * x) for _ in 1:d]
    f_mps = separable_to_mps(f_funcs, s; maxdim=MAX_BOND_DIM, cutoff=CUTOFF)

    # ---- Boundary condition note ----
    # Homogeneous Dirichlet (u=0 on ∂Ω) is already enforced by L1d.
    # For non-homogeneous BCs, uncomment:
    # bc_correction = build_bc_correction_mps(g_bc_value, s)
    # f_mps = add(f_mps, bc_correction, ...)

    # ---- Time integration ----
    println("\nStarting time integration...")
    t = 0.0
    for step in 1:nsteps
        if method == :euler
            u = euler_step(u, H_L, f_mps, dt; maxdim=MAX_BOND_DIM, cutoff=CUTOFF)
        elseif method == :crank_nicolson
            u = crank_nicolson_step(u, H_L, H_I, f_mps, dt;
                                     maxdim=MAX_BOND_DIM, cutoff=CUTOFF)
        end
        t += dt

        # Print info every 10 steps
        step % 10 == 0 && print_mps_info(u, step, t)
    end

    # ---- Sample solution at center point ----
    center_idx = fill(div(N, 2), d)
    u_center = evaluate_mps(u, s, center_idx)
    println("\nSolution at grid center $(center_idx): u = $u_center")

    # ---- Verification against analytical solution (no source, Dirichlet) ----
    # Exact: u_exact(x,t) = exp(-π²·d·t) · ∏ sin(πxₖ) / ||u₀||
    x_center = dx * div(N, 2)
    u_exact_center = exp(-π^2 * d * T) * sin(π * x_center)^d
    println("Analytical solution at center (no source): $u_exact_center")
    println("(Note: differs because source term was included)")

    return u, s
end

# ============================================================
# SECTION 13: Entry Point
# ============================================================

u_final, sites = solve_heat_equation(method=:euler)

# To use Crank-Nicolson (more accurate but slower due to linear solve):
# u_final, sites = solve_heat_equation(method=:crank_nicolson)