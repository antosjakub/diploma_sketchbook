# ═══════════════════════════════════════════════════════════════════════════
#  mlmc.jl — Multilevel Monte Carlo for Feynman-Kac (Giles 2008)
#
#  Key idea: instead of one fine-resolution MC estimate, write
#      𝔼[P_L] = 𝔼[P₀] + Σₗ₌₁ᴸ 𝔼[Pₗ − Pₗ₋₁]
#  where Pₗ uses time step hₗ = T·M⁻ˡ.  Couple fine/coarse paths with
#  the SAME Brownian increments so Var(Pₗ − Pₗ₋₁) → 0, requiring fewer
#  samples at finer levels.  Total cost ≈ O(ε⁻²) vs O(ε⁻²·|log ε|²)
#  for standard MC with comparable bias.
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  Single-level Euler-Maruyama batch (used internally)
#  Returns vector of path functionals Φ (length N)
# ---------------------------------------------------------------------------
function _em_level(ic, x_d, t_eval, n_steps, Nh,
                   drift_fn, drift_const, sigma,
                   potential, source,
                   gpu, ::Type{FT}) where FT
    d  = size(x_d, 1)
    dt = FT(t_eval / n_steps)
    sq = FT(sqrt(t_eval / n_steps))
    σ_f = FT(sigma)

    X     = repeat(x_d, 1, Nh)
    V_acc = dev_zeros(FT, Nh; gpu)
    f_acc = dev_zeros(FT, Nh; gpu)

    has_V = potential !== nothing
    has_f = source    !== nothing

    for k in 0:n_steps-1
        tk    = FT(k * Float64(dt))
        t_pde = FT(t_eval) - tk
        dW    = dev_randn(FT, d, Nh; gpu) .* sq

        if has_f
            fv = source(t_pde, X)
            f_acc .= f_acc .+ fv .* exp.(.-V_acc) .* dt
        end
        if has_V
            V_acc .= V_acc .+ potential(tk, X) .* dt
        end

        if drift_fn !== nothing
            μk = drift_fn(tk, X)
        else
            μk = drift_const
        end
        X .= X .+ μk .* dt .+ σ_f .* dW
    end

    return vec(ic(X) .* exp.(.-V_acc) .+ f_acc)
end

# ---------------------------------------------------------------------------
#  Coupled fine/coarse batch — returns vector of (P_fine − P_coarse)
#  Fine: n_steps_fine time steps.   Coarse: n_steps_fine / M steps.
#  Same Brownian path: sum M fine increments → one coarse increment.
# ---------------------------------------------------------------------------
function _coupled_level(ic, x_d, t_eval, n_fine, M, Nh,
                        drift_fn, drift_const, sigma,
                        potential, source,
                        gpu, ::Type{FT}) where FT
    d  = size(x_d, 1)
    n_coarse = n_fine ÷ M
    dt_f = FT(t_eval / n_fine)
    dt_c = FT(t_eval / n_coarse)
    sq_f = FT(sqrt(t_eval / n_fine))
    σ_f  = FT(sigma)

    Xf     = repeat(x_d, 1, Nh)
    Xc     = copy(Xf)
    Vf_acc = dev_zeros(FT, Nh; gpu)
    Vc_acc = dev_zeros(FT, Nh; gpu)
    ff_acc = dev_zeros(FT, Nh; gpu)
    fc_acc = dev_zeros(FT, Nh; gpu)

    has_V = potential !== nothing
    has_f = source    !== nothing

    for kc in 0:n_coarse-1
        # Accumulate M fine steps, tracking the summed Brownian increment
        dW_sum = dev_zeros(FT, d, Nh; gpu)

        for m in 0:M-1
            kf    = kc * M + m
            tk_f  = FT(kf * Float64(dt_f))
            t_pde = FT(t_eval) - tk_f
            dW    = dev_randn(FT, d, Nh; gpu) .* sq_f
            dW_sum .= dW_sum .+ dW

            # Fine path step
            if has_f
                fv = source(t_pde, Xf)
                ff_acc .= ff_acc .+ fv .* exp.(.-Vf_acc) .* dt_f
            end
            if has_V
                Vf_acc .= Vf_acc .+ potential(tk_f, Xf) .* dt_f
            end

            if drift_fn !== nothing
                μk = drift_fn(tk_f, Xf)
            else
                μk = drift_const
            end
            Xf .= Xf .+ μk .* dt_f .+ σ_f .* dW
        end

        # Coarse path step (one big step using summed increments)
        tk_c  = FT(kc * Float64(dt_c))
        t_pde_c = FT(t_eval) - tk_c

        if has_f
            fv = source(t_pde_c, Xc)
            fc_acc .= fc_acc .+ fv .* exp.(.-Vc_acc) .* dt_c
        end
        if has_V
            Vc_acc .= Vc_acc .+ potential(tk_c, Xc) .* dt_c
        end

        if drift_fn !== nothing
            μk = drift_fn(tk_c, Xc)
        else
            μk = drift_const
        end
        Xc .= Xc .+ μk .* dt_c .+ σ_f .* dW_sum
    end

    Pf = vec(ic(Xf) .* exp.(.-Vf_acc) .+ ff_acc)
    Pc = vec(ic(Xc) .* exp.(.-Vc_acc) .+ fc_acc)
    return Pf .- Pc
end


# ---------------------------------------------------------------------------
#  solve_mlmc — adaptive MLMC (Giles 2008, Algorithm 1)
# ---------------------------------------------------------------------------
"""
    solve_mlmc(ic, x, t_eval;
               drift = zeros(length(x)), drift_fn = nothing, sigma = 1.0,
               potential = nothing, source = nothing,
               target_rmse = 1e-3, M = 2, L_min = 2, L_max = 8,
               N_pilot = 5000, gpu = HAS_GPU[], FT = ...)

Multilevel Monte Carlo Feynman-Kac solver.  Automatically selects the number
of levels and samples per level to achieve `target_rmse` (root mean square error)
at near-optimal cost.

Returns `FKResult`.
"""
function solve_mlmc(ic, x::Vector{Float64}, t_eval::Float64;
                    drift::Vector{Float64} = zeros(length(x)),
                    drift_fn = nothing,
                    sigma::Float64 = 1.0,
                    potential = nothing,
                    source = nothing,
                    target_rmse::Float64 = 1e-3,
                    M::Int = 2,
                    L_min::Int = 2,
                    L_max::Int = 8,
                    N_pilot::Int = 5000,
                    gpu::Bool = HAS_GPU[],
                    FT::Type{<:AbstractFloat} = gpu ? Float32 : Float64)

    t0 = time()
    d  = length(x)
    ε  = target_rmse
    x_d = to_dev(reshape(FT.(x), d, 1), Val(gpu))

    # Constant drift on device (used when drift_fn is nothing)
    μ_const = to_dev(reshape(FT.(drift), d, 1), Val(gpu))
    dc = drift_fn === nothing ? μ_const : nothing

    # Storage for level statistics
    sums   = Float64[]     # Σ Yₗ
    sumsq  = Float64[]     # Σ Yₗ²
    Nl     = Int[]         # samples taken at each level
    costl  = Float64[]     # cost per sample at each level (∝ n_steps)

    # ── Phase 1: pilot runs to estimate variance at each level ─────────
    L = L_min
    for l in 0:L
        n_steps_f = M^(l + 1)     # fine steps (level 0 uses M steps, not 1)
        Np = N_pilot

        if l == 0
            vals = Array(_em_level(ic, x_d, t_eval, n_steps_f, Np,
                                   drift_fn, dc, sigma, potential, source, gpu, FT))
        else
            vals = Array(_coupled_level(ic, x_d, t_eval, n_steps_f, M, Np,
                                        drift_fn, dc, sigma, potential, source, gpu, FT))
        end

        push!(sums,  sum(vals))
        push!(sumsq, sum(vals .^ 2))
        push!(Nl,    Np)
        push!(costl, Float64(n_steps_f))
    end

    # ── Phase 2: iterative refinement ──────────────────────────────────
    for iter in 1:20
        # Compute variance estimates
        Vl = [(sumsq[l+1] / Nl[l+1]) - (sums[l+1] / Nl[l+1])^2 for l in 0:L]
        Vl = max.(Vl, 1e-30)   # avoid zero variance

        # Optimal sample counts (Giles formula)
        Cl = costl[1:L+1]
        vc_sum = sum(sqrt(Vl[l+1] * Cl[l+1]) for l in 0:L)
        N_opt  = [max(N_pilot, ceil(Int, 2.0 * ε^(-2) * sqrt(Vl[l+1] / Cl[l+1]) * vc_sum))
                  for l in 0:L]

        # Run additional samples where needed
        did_work = false
        for l in 0:L
            ΔN = N_opt[l+1] - Nl[l+1]
            ΔN <= 0 && continue
            did_work = true

            n_steps_f = M^(l + 1)
            if l == 0
                vals = Array(_em_level(ic, x_d, t_eval, n_steps_f, ΔN,
                                       drift_fn, dc, sigma, potential, source, gpu, FT))
            else
                vals = Array(_coupled_level(ic, x_d, t_eval, n_steps_f, M, ΔN,
                                            drift_fn, dc, sigma, potential, source, gpu, FT))
            end

            sums[l+1]  += sum(vals)
            sumsq[l+1] += sum(vals .^ 2)
            Nl[l+1]    += ΔN
        end

        # Check bias: if the finest-level correction is still large, add a level
        mean_L = abs(sums[end] / Nl[end])
        if L < L_max && mean_L > ε / sqrt(2.0) * (M - 1)
            L += 1
            n_steps_f = M^(L + 1)
            vals = Array(_coupled_level(ic, x_d, t_eval, n_steps_f, M, N_pilot,
                               drift_fn, dc, sigma, potential, source, gpu, FT))
            push!(sums,  sum(vals))
            push!(sumsq, sum(vals .^ 2))
            push!(Nl,    N_pilot)
            push!(costl, Float64(n_steps_f))
            continue
        end

        !did_work && break
    end

    # ── Result ─────────────────────────────────────────────────────────
    μ_est = sum(sums[l+1] / Nl[l+1] for l in 0:L)

    # Variance of the estimator
    Vl = [(sumsq[l+1] / Nl[l+1]) - (sums[l+1] / Nl[l+1])^2 for l in 0:L]
    σ_est = sqrt(sum(max(0.0, Vl[l+1]) / Nl[l+1] for l in 0:L))

    total_N = sum(Nl)

    @info @sprintf("MLMC: L=%d, levels N=%s, total=%d, est=%.6e ± %.2e",
                   L, string(Nl), total_N, μ_est, σ_est)

    return FKResult(μ_est, σ_est, total_N, time() - t0)
end
