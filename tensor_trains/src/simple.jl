# Minimal d-dimensional heat equation solver using ITensors.jl (MPS/MPO)
#
# Solves:  ∂u/∂t = Δu + f,   u=0 on boundary,   u(x,0) = u₀(x)
# Method:  Explicit Euler, homogeneous Dirichlet BCs, separable IC & source
#
# Install deps:
#   using Pkg; Pkg.add(["ITensors", "ITensorMPS"])

using ITensors
using ITensorMPS
using LinearAlgebra

# --- Parameters ---
const d   = 8
const N   = 32
const dx  = 1.0 / (N + 1)
const dt  = 1e-6       # must satisfy dt < dx²/(2d) ≈ 5.7e-6 — adjust if needed
const T   = 1e-3
const MAXDIM = 32
const CUTOFF = 1e-10

# --- 1D Laplacian (finite differences, Dirichlet BCs) ---
function make_L1d()
    L = zeros(N, N)
    for i in 1:N
        L[i, i] = -2.0 / dx^2
        i > 1 && (L[i, i-1] = 1.0 / dx^2)
        i < N && (L[i, i+1] = 1.0 / dx^2)
    end
    return L
end

# --- Laplacian MPO using Kronecker sum structure: Δ = Σₖ I⊗…⊗L₁ᵈ⊗…⊗I ---
# Bond dimension = 2, using a 2-state FSM: state 1 = "done", state 2 = "not yet"
function make_laplacian_mpo(sites, L1d)
    Id = Matrix{Float64}(I, N, N)
    links = [Index(2, "Link,l=$l") for l in 1:d-1]
    tensors = Vector{ITensor}(undef, d)

    for k in 1:d
        s, sp = sites[k], sites[k]'
        lL = k > 1 ? links[k-1] : nothing
        lR = k < d ? links[k]   : nothing

        inds_W = filter(!isnothing, [sp, s, lL, lR])
        W = ITensor(0.0, inds_W...)

        for i in 1:N, j in 1:N
            function set!(val, states...)
                all_inds = filter(!isnothing, [sp=>i, s=>j,
                    !isnothing(lL) ? lL=>states[1] : nothing,
                    !isnothing(lR) ? lR=>states[end] : nothing])
                W[all_inds...] += val
            end

            if k == 1          # left boundary: no lL
                L1d[i,j] != 0 && (W[sp=>i, s=>j, lR=>1] += L1d[i,j])  # act here
                i == j         && (W[sp=>i, s=>j, lR=>2] += Id[i,j])   # act later
            elseif k == d      # right boundary: no lR
                i == j         && (W[sp=>i, s=>j, lL=>1] += Id[i,j])   # already done
                L1d[i,j] != 0 && (W[sp=>i, s=>j, lL=>2] += L1d[i,j])  # act here
            else               # interior
                i == j         && (W[sp=>i, s=>j, lL=>1, lR=>1] += 1.0) # pass through
                L1d[i,j] != 0 && (W[sp=>i, s=>j, lL=>2, lR=>1] += L1d[i,j]) # act
                i == j         && (W[sp=>i, s=>j, lL=>2, lR=>2] += 1.0) # defer
            end
        end
        tensors[k] = W
    end
    return MPO(tensors)
end

# --- Build MPS from separable function u(x₁,...,xₐ) = ∏ fₖ(xₖ) ---
function separable_mps(funcs, sites)
    x = [i * dx for i in 1:N]
    links = [Index(1, "Link,l=$l") for l in 1:d-1]
    tensors = Vector{ITensor}(undef, d)
    for k in 1:d
        if k == 1
            T = ITensor(sites[k], links[k])
            for i in 1:N; T[sites[k]=>i, links[k]=>1] = funcs[k](x[i]); end
        elseif k == d
            T = ITensor(sites[k], links[k-1])
            for i in 1:N; T[sites[k]=>i, links[k-1]=>1] = funcs[k](x[i]); end
        else
            T = ITensor(sites[k], links[k-1], links[k])
            for i in 1:N; T[sites[k]=>i, links[k-1]=>1, links[k]=>1] = funcs[k](x[i]); end
        end
        tensors[k] = T
    end
    return MPS(tensors)
end

# --- Evaluate MPS at a grid multi-index ---
function eval_mps(u, sites, idx)
    v = ITensor(1.0)
    for k in 1:d
        v = v * u[k] * onehot(sites[k] => idx[k])
    end
    return scalar(v)
end

# --- Main ---
function main()
    sites = [Index(N, "Site,n=$k") for k in 1:d]
    L1d   = make_L1d()
    H_L   = make_laplacian_mpo(sites, L1d)

    # Initial condition: u₀ = ∏ sin(πxₖ)
    u = separable_mps([x -> sin(π*x) for _ in 1:d], sites)

    # Source term: f = ∏ sin(πxₖ)  (constant in time)
    f = separable_mps([x -> sin(π*x) for _ in 1:d], sites)

    # Time loop: u ← u + dt*(L*u + f)
    println("Running $( round(Int, T/dt) ) steps...")
    t = 0.0
    for step in 1:round(Int, T/dt)
        Lu    = apply(H_L, u; maxdim=MAXDIM, cutoff=CUTOFF)
        #u     = add(add(u, Lu, 1.0, dt), f, 1.0, dt; maxdim=MAXDIM, cutoff=CUTOFF)
        du = add(Lu, f; maxdim=MAXDIM, cutoff=CUTOFF)
        u  = add(u, dt * du; maxdim=MAXDIM, cutoff=CUTOFF)
        t += dt
        step % 20 == 0 && println("t=$(round(t,digits=5))  ||u||=$(round(norm(u),sigdigits=4))  max_bond=$(maxlinkdim(u))")
    end

    # Sample solution at grid center
    ctr = fill(div(N,2), d)
    println("\nu at center: $(eval_mps(u, sites, ctr))")
    println("Analytical (no source): $(exp(-π^2 * d * t) * sin(π * div(N,2) * dx)^d)")
end

main()