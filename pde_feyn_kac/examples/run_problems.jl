# ═══════════════════════════════════════════════════════════════════════════
#  run_problems.jl — Feynman-Kac solvers for all four problems from problem_sets.md
#
#  Usage:  julia --project=. examples/run_problems.jl
#
#  Each problem maps the PDE to the FK convention:
#      PDE:  ∂ₜu = μ·∇u + ½σ²Δu − Vu + f,   u(0,x) = g(x)
#      SDE:  dXₛ = μ ds + σ dWₛ,  X₀ = x
#      FK:   u(t,x) = 𝔼ₓ[ g(Xₜ) e^{−∫V} + ∫ f(t−s,Xₛ) e^{−∫V} ds ]
# ═══════════════════════════════════════════════════════════════════════════

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "FeynmanKac.jl"))
using .FeynmanKac
using Printf, Random


# ═══════════════════════════════════════════════════════════════════════════
#  Problem 1:  Product of sines (pure diffusion, V=0, f=0)
#
#  PDE:   ∂ₜu = α Δu          (forward heat equation)
#  SDE:   dX = √(2α) dW       (pure Brownian motion, no drift)
#  IC:    g(x) = ∏ sin(aᵢ xᵢ)
#  Exact: u(t,x) = [∏ sin(aᵢ xᵢ)] exp(−α|a|²t)
# ═══════════════════════════════════════════════════════════════════════════
function run_product_of_sines(; d::Int=10, N::Int=10^6)
    println("\n" * "="^70)
    println("  Problem 1: Product of sines  (d=$d)")
    println("="^70)

    α = 0.1
    a = ones(d)                                     # wave numbers
    x = fill(1.0, d)                                # evaluation point
    t_eval = 0.5
    σ_sde = sqrt(2α)

    # IC: g(X) = ∏ sin(aᵢ Xᵢ)  —  operates on (d, N) arrays
    ic(X) = vec(prod(sin.(X), dims=1))              # a=1 so sin(aᵢXᵢ)=sin(Xᵢ)

    # Analytic solution
    exact = prod(sin.(a .* x)) * exp(-α * sum(a .^ 2) * t_eval)

    # ── Exact solver (no time-stepping, single Gaussian sample) ──
    res = solve_mc(ic, x, t_eval, N; sigma=σ_sde)
    @printf("  Exact MC:  u = %+.8f  (exact = %+.8f,  err = %.2e,  σ = %.2e,  %.3fs)\n",
            res.value, exact, abs(res.value - exact), res.std_error, res.elapsed)

    return res
end


# ═══════════════════════════════════════════════════════════════════════════
#  Problem 2:  Gauss diffusion with advection (V=0, f=0, constant drift)
#
#  PDE:   ∂ₜu = −a (v·∇u) − b Δu   ... SIGN NOTE: the problem states
#         ∂ₜu + a∇u·v + bΔu = 0,  which is ∂ₜu = −a v·∇u − bΔu.
#         For this to be a well-posed forward problem, we interpret it as
#         the backward Kolmogorov eq. of the SDE dX = a·v dt + √(2b) dW.
#         Then u(t,x) = 𝔼[g(Xₜ) | X₀=x].
#
#  SDE:   dX = a·v dt + √(2b) dW
#  IC:    g(x) = (2π)^{−d/2} exp(−|x|²/2)
#  No closed-form in general, but FK gives an analytic Gaussian convolution.
# ═══════════════════════════════════════════════════════════════════════════
function run_gauss_diffusion(; d::Int=10, N::Int=10^6)
    println("\n" * "="^70)
    println("  Problem 2: Gauss diffusion  (d=$d)")
    println("="^70)

    a_coeff = 0.5
    b_coeff = 0.1
    v = ones(d) / sqrt(d)                           # unit vector
    x = zeros(d)                                    # evaluate at origin
    t_eval = 0.5

    μ_sde = a_coeff .* v
    σ_sde = sqrt(2b_coeff)

    # IC: g(x) = (2π)^{-d/2} exp(-|x|²/2)
    function ic(X)
        # X is (d, N)
        sq = vec(sum(X .^ 2, dims=1))
        return exp.(sq .* eltype(X)(-0.5)) .* eltype(X)((2π)^(-d / 2))
    end

    # Analytic: u(t,x) = (2π)^{-d/2} (1+2bt)^{-d/2} exp(-|x-avt|²/(2(1+2bt)))
    s2 = 1 + 2b_coeff * t_eval
    shift = x .- a_coeff .* v .* t_eval
    exact = (2π)^(-d / 2) * s2^(-d / 2) * exp(-sum(shift .^ 2) / (2s2))

    res = solve_mc(ic, x, t_eval, N; drift=μ_sde, sigma=σ_sde)
    @printf("  Exact MC:  u = %.8e  (exact = %.8e,  err = %.2e,  σ = %.2e,  %.3fs)\n",
            res.value, exact, abs(res.value - exact), res.std_error, res.elapsed)

    return res
end


# ═══════════════════════════════════════════════════════════════════════════
#  Problem 3:  Travelling Gaussian packet (V≠0, f≠0)
#
#  PDE:  ∂ₜu − δΔu + v·∇u + w·u = f
#
#  With parameters (matching the verified PyTorch implementation):
#     δ = 1.0  (diffusion)
#     v = −c/a
#     w = −2δα Σaᵢ²     ← NOTE: w is NEGATIVE (problem_sets.md omits the minus)
#     f = (−4α²δ Σ(aᵢzᵢ)² − β) cos(γt) e^{…} − γ sin(γt) e^{…}
#
#  FK mapping: ∂ₜu = δΔu − v·∇u − w·u + f
#     ⟹  μ = −v,  σ = √(2δ),  V = w  (w<0 means growth),  source = f
#
#  IC:    u(0,x) = exp(−α Σ(aᵢxᵢ − bᵢ)²)
#  Exact: u(t,x) = exp(−α Σ(aᵢxᵢ − bᵢ + cᵢt)²) exp(−βt) cos(γt)
# ═══════════════════════════════════════════════════════════════════════════
function run_travelling_gaussian(; d::Int=10, N::Int=10^6, n_steps::Int=200)
    println("\n" * "="^70)
    println("  Problem 3: Travelling Gaussian  (d=$d)")
    println("="^70)

    # Parameters (matching the PyTorch TravellingGaussPacket_v2)
    δ = 1.0                                          # diffusion (δ=1.0 in PyTorch)
    α = 1.0
    β = 0.1
    γ = 2.0
    a_vec = ones(d)
    b_vec = zeros(d)
    c_vec = fill(0.5, d)
    w = -2δ * α * sum(a_vec .^ 2)                   # w < 0 !

    # SDE coefficients
    v_drift = -c_vec ./ a_vec                        # vᵢ = −cᵢ/aᵢ
    μ_sde   = -v_drift                               # SDE drift = −v
    σ_sde   = sqrt(2δ)

    x = fill(0.5, d)
    t_eval = 0.3

    # IC: g(x) = exp(−α Σ (aᵢxᵢ − bᵢ)²)
    function ic(X)
        FT = eltype(X)
        a_d = to_dev_like(FT.(a_vec), X)
        b_d = to_dev_like(FT.(b_vec), X)
        diff = a_d .* X .- b_d
        return vec(exp.(FT(-α) .* sum(diff .^ 2, dims=1)))
    end

    # Potential: V = w  (w is already negative, so exp(−∫V) = exp(|w|t) > 1)
    potential(t, X) = fill!(similar(X, size(X, 2)), eltype(X)(w))

    # Source: f = (−4α²δ Σ(aᵢzᵢ)² − β) cos(γt) e^{…} − γ sin(γt) e^{…}
    # Matching PyTorch: f_sim_inner = -4α²δ Σ(aᵢzᵢ)²
    #                   f = (f_sim_inner − β) cos(γt) e^{…} − γ sin(γt) e^{…}
    function source_fn(t_pde, X)
        FT = eltype(X)
        a_d = to_dev_like(FT.(a_vec), X)
        b_d = to_dev_like(FT.(b_vec), X)
        c_d = to_dev_like(FT.(c_vec), X)

        z = a_d .* X .- b_d .+ c_d .* t_pde
        f_sim_inner = FT(-4α^2 * δ) .* sum((a_d .* z) .^ 2, dims=1)
        gauss = exp.(FT(-α) .* sum(z .^ 2, dims=1) .- FT(β * Float64(t_pde)))

        bracket = (f_sim_inner .- FT(β)) .* FT(cos(γ * Float64(t_pde))) .-
                  FT(γ * sin(γ * Float64(t_pde)))
        return vec(bracket .* gauss)
    end

    # Exact solution
    z_exact = a_vec .* x .- b_vec .+ c_vec .* t_eval
    exact = exp(-α * sum(z_exact .^ 2)) * exp(-β * t_eval) * cos(γ * t_eval)

    res = solve_fk(ic, x, t_eval, N, n_steps;
                   drift=μ_sde, sigma=σ_sde,
                   potential=potential, source=source_fn)
    @printf("  EM MC:     u = %+.8f  (exact = %+.8f,  err = %.2e,  σ = %.2e,  %.3fs)\n",
            res.value, exact, abs(res.value - exact), res.std_error, res.elapsed)

    # Also try MLMC
    println("  Running MLMC...")
    res_mlmc = solve_mlmc(ic, x, t_eval;
                          drift=μ_sde, sigma=σ_sde,
                          potential=potential, source=source_fn,
                          target_rmse=1e-3)
    @printf("  MLMC:      u = %+.8f  (exact = %+.8f,  err = %.2e,  σ = %.2e,  %.3fs)\n",
            res_mlmc.value, exact, abs(res_mlmc.value - exact), res_mlmc.std_error, res_mlmc.elapsed)

    # ── Importance sampling via Girsanov ──────────────────────────────────
    # Optimal IS drift: θ* = σ ∇log u.  Since u is known analytically:
    #   log u = −α Σ(aᵢxᵢ − bᵢ + cᵢτ)² − βτ + log cos(γτ)
    #   ∇ₓ log u = −2α aᵢ(aᵢxᵢ − bᵢ + cᵢτ)   (component i)
    #   θ* = σ · ∇log u
    # where τ = t_eval − t_sde is the PDE time.
    println("  Running EM MC + importance sampling...")
    function is_drift_fn(t_sde, X)
        FT = eltype(X)
        a_d = to_dev_like(FT.(a_vec), X)
        b_d = to_dev_like(FT.(b_vec), X)
        c_d = to_dev_like(FT.(c_vec), X)
        τ = FT(t_eval) - t_sde                         # PDE time
        z = a_d .* X .- b_d .+ c_d .* τ
        return FT(σ_sde) .* (FT(-2α) .* a_d .* z)      # σ · ∇log u
    end

    res_is = solve_fk(ic, x, t_eval, N, n_steps;
                      drift=μ_sde, sigma=σ_sde,
                      potential=potential, source=source_fn,
                      is_drift=is_drift_fn)
    @printf("  EM MC+IS:  u = %+.8f  (exact = %+.8f,  err = %.2e,  σ = %.2e,  %.3fs)\n",
            res_is.value, exact, abs(res_is.value - exact), res_is.std_error, res_is.elapsed)
    @printf("  Variance reduction factor: %.1fx\n", res.std_error / max(res_is.std_error, 1e-30))

    return res
end


# ═══════════════════════════════════════════════════════════════════════════
#  Problem 4:  Fokker-Planck / Langevin (7-atom molecule, d=21)
#
#  SDE:  dX = −(1/ξ)∇U(X) dt + √(2D) dW
#
#  This example uses a PLACEHOLDER Lennard-Jones potential.
#  Replace grad_U and laplace_U with your actual interatomic potential.
#
#  Two modes:
#    (a) Solve the backward Kolmogorov PDE for 𝔼[g(Xₜ)] via MLMC
#    (b) Estimate mean first-passage time between configurations
# ═══════════════════════════════════════════════════════════════════════════

# ── Placeholder: Lennard-Jones pairwise potential for 7 atoms in 3D ──
# Replace these with your actual potential functions.
function _lj_grad!(X_mat)
    # X_mat is (21, N) — reshape to (3, 7, N) for atom coordinates
    FT = eltype(X_mat)
    d, N = size(X_mat)
    n_atoms = d ÷ 3
    grad = similar(X_mat)
    fill!(grad, FT(0))

    # Reshape for convenience: access atom i coords as X[3(i-1)+1 : 3i, :]
    ε = FT(1.0)   # LJ energy scale
    σ_lj = FT(1.0)   # LJ length scale

    for i in 1:n_atoms
        ri = (3(i-1)+1):(3i)
        for j in (i+1):n_atoms
            rj = (3(j-1)+1):(3j)
            dx = X_mat[ri, :] .- X_mat[rj, :]         # (3, N)
            r2 = sum(dx .^ 2, dims=1) .+ FT(1e-8)     # (1, N), avoid /0
            inv_r2 = FT(1) ./ r2
            inv_r6 = inv_r2 .^ 3
            # F = 24ε (2σ¹²/r¹³ − σ⁶/r⁷) r̂  = 24ε (2·inv_r6² − inv_r6)·inv_r2 · dx
            fmag = FT(24) .* ε .* (FT(2) .* inv_r6 .^ 2 .- inv_r6) .* inv_r2
            fvec = fmag .* dx                           # (3, N)
            grad[ri, :] .= grad[ri, :] .- fvec          # ∇U on atom i
            grad[rj, :] .= grad[rj, :] .+ fvec          # ∇U on atom j
        end
    end
    return grad
end

function run_fokker_planck(; d::Int=21, N::Int=10^5, mode::Symbol=:fpt)
    println("\n" * "="^70)
    println("  Problem 4: Fokker-Planck / Langevin  (d=$d, mode=$mode)")
    println("="^70)

    ξ = 1.0     # friction
    D = 0.1     # diffusion constant
    σ_sde = sqrt(2D)

    if mode == :backward_kolmogorov
        # Solve ∂ₜu = −(1/ξ)∇U·∇u + DΔu  via MLMC
        # IC: g(x) = exp(−|x|²) (example observable)
        x = randn(d) .* 0.5 .+ 1.0                  # random initial config
        t_eval = 1.0

        ic(X) = vec(exp.(eltype(X)(-1) .* sum(X .^ 2, dims=1)))

        drift_fn(t, X) = eltype(X)(-1.0 / ξ) .* _lj_grad!(X)

        println("  Running MLMC for backward Kolmogorov equation...")
        res = solve_mlmc(ic, x, t_eval;
                         drift_fn=drift_fn, sigma=σ_sde,
                         target_rmse=1e-2, N_pilot=2000)
        @printf("  MLMC:  u = %.6e ± %.2e  (%.3fs)\n",
                res.value, res.std_error, res.elapsed)
        return res

    elseif mode == :fpt
        # First-passage time from config A to neighborhood of config B
        # Place 7 atoms in two different configurations
        x0 = vcat([Float64[i, 0, 0] for i in 1:7]...)       # linear chain
        target = vcat([Float64[cos(2π*i/7), sin(2π*i/7), 0.0] for i in 1:7]...)  # ring

        grad_U(X) = _lj_grad!(X)
        target_radius = 2.0                           # convergence ball radius

        println("  Estimating mean first-passage time...")
        println("  x0:     linear chain")
        println("  target: ring (radius=$target_radius)")

        res = estimate_fpt(grad_U, x0, target, target_radius, N;
                           xi=ξ, D=D, dt=5e-3, max_steps=200_000)
        if !isnan(res.value)
            @printf("  FPT:   τ = %.4f ± %.4f  (%.3fs)\n",
                    res.value, res.std_error, res.elapsed)
        end
        return res
    end
end


# ═══════════════════════════════════════════════════════════════════════════
#  Scalability test — push dimension as high as possible
# ═══════════════════════════════════════════════════════════════════════════
function run_scaling_test(; dims=[3, 10, 21, 50, 100], N::Int=10^6)
    println("\n" * "="^70)
    println("  Scaling test: product-of-sines across dimensions")
    println("="^70)
    println("  d       | exact MC value  | exact         | abs error    | time")
    println("  --------+-----------------+---------------+--------------+-------")

    α = 0.1
    t_eval = 0.1   # short time to keep solution from being too small

    for d in dims
        x = fill(π / 4, d)           # sin(π/4) = √2/2 in each dim
        σ_sde = sqrt(2α)
        exact = (sin(π / 4))^d * exp(-α * d * t_eval)

        ic(X) = vec(prod(sin.(X), dims=1))

        res = solve_mc(ic, x, t_eval, N; sigma=σ_sde)
        @printf("  d=%-5d | %+.8e | %+.8e | %.2e | %.3fs\n",
                d, res.value, exact, abs(res.value - exact), res.elapsed)
    end
end


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-point demo — shared Brownian increments via solve_fk_multi
#
#  Evaluates product-of-sines at P nearby points.  Shared dW gives the same
#  per-point accuracy as independent runs, but differences u(x₁)−u(x₂)
#  between nearby points become much more precise because the correlated
#  noise cancels in the subtraction.
# ═══════════════════════════════════════════════════════════════════════════
function run_multi_point_demo(; d::Int=5, P::Int=5, N::Int=200_000, n_steps::Int=50)
    println("\n" * "="^70)
    println("  Multi-point demo: shared vs independent Brownian increments  (d=$d, P=$P)")
    println("="^70)

    α = 0.1
    t_eval = 0.5
    σ_sde = sqrt(2α)

    ic(X) = vec(prod(sin.(X), dims=1))
    exact_at(xp) = prod(sin.(xp)) * exp(-α * d * t_eval)

    # P nearby points: small perturbation around x = [1, …, 1]
    Random.seed!(42)
    x_center = ones(d)
    xs = x_center .+ randn(d, P) .* 0.05      # (d, P)

    # ── Shared dW (solve_fk_multi) ────────────────────────────────────────
    results_shared = solve_fk_multi(ic, xs, t_eval, N, n_steps; sigma=σ_sde)

    println("\n  Per-point estimates (shared dW):")
    for p in 1:P
        ex = exact_at(xs[:, p])
        r  = results_shared[p]
        @printf("    x%d: u=%+.8f  exact=%+.8f  err=%.2e  σ=%.2e\n",
                p, r.value, ex, abs(r.value - ex), r.std_error)
    end

    # ── Independent dW (separate solve_fk per point) ──────────────────────
    results_indep = [solve_fk(ic, xs[:, p], t_eval, N, n_steps; sigma=σ_sde)
                     for p in 1:P]

    println("\n  Per-point estimates (independent dW):")
    for p in 1:P
        ex = exact_at(xs[:, p])
        r  = results_indep[p]
        @printf("    x%d: u=%+.8f  exact=%+.8f  err=%.2e  σ=%.2e\n",
                p, r.value, ex, abs(r.value - ex), r.std_error)
    end

    # ── Compare pairwise differences ──────────────────────────────────────
    println("\n  Pairwise differences u(x₁) − u(xₚ):")
    println("    pair  | exact diff   | shared dW    | indep dW     | shared err   | indep err")
    println("    ------+--------------+--------------+--------------+--------------+-------------")
    for p in 2:P
        ex_diff    = exact_at(xs[:, 1]) - exact_at(xs[:, p])
        diff_sh    = results_shared[1].value - results_shared[p].value
        diff_ind   = results_indep[1].value  - results_indep[p].value
        err_sh     = abs(diff_sh - ex_diff)
        err_ind    = abs(diff_ind - ex_diff)
        @printf("    1−%-3d | %+.6e | %+.6e | %+.6e | %.2e | %.2e\n",
                p, ex_diff, diff_sh, diff_ind, err_sh, err_ind)
    end

    @printf("\n  Shared-dW time:  %.3fs for %d points\n", results_shared[1].elapsed, P)
    @printf("  Independent time: %.3fs for %d points (sum of %d individual runs)\n",
            sum(r.elapsed for r in results_indep), P, P)

    return results_shared
end


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
function main()
    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║  Feynman-Kac PDE Solver — GPU-Accelerated Monte Carlo            ║")
    println("╚══════════════════════════════════════════════════════════════════╝")

    run_product_of_sines(d=10)
    run_gauss_diffusion(d=10)
    run_travelling_gaussian(d=10, n_steps=200)
    run_fokker_planck(d=21, mode=:backward_kolmogorov)
    run_multi_point_demo(d=5, P=5)
    run_scaling_test()
end

main()
