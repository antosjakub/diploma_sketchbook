# ═══════════════════════════════════════════════════════════════════════════
#  core.jl — Monte Carlo and Euler-Maruyama solvers for the Feynman-Kac formula
#
#  Convention (initial-value problem):
#
#    PDE:   ∂ₜu = μ·∇u + ½σ²Δu − V u + f,    u(0,x) = g(x)
#
#    SDE:   dXₛ = μ(s, Xₛ)ds + σ(s, Xₛ)dWₛ,   X₀ = x
#
#    FK:    u(t,x) = 𝔼ₓ[ g(Xₜ) e^{−∫₀ᵗ V(Xₛ)ds}
#                       + ∫₀ᵗ f(t−s, Xₛ) e^{−∫₀ˢ V(Xᵣ)dr} ds ]
#
#  All user-supplied functions operate on *batched* data:
#    - X  is a (d, N) matrix   (each column = one path position)
#    - Return values are length-N vectors (one scalar per path)
#    - Works identically on CuArray (GPU) or Array (CPU)
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  solve_mc — simple Monte Carlo for V=0, f=0
#
#  For CONSTANT drift μ and SCALAR diffusion σ the SDE is solved exactly
#  in a single step (no time-stepping error):
#      Xₜ = x + μt + σ√t Z,   Z ~ N(0,Iₐ)
#
#  For POSITION-DEPENDENT drift, Euler-Maruyama with n_steps is used.
# ---------------------------------------------------------------------------
"""
    solve_mc(ic, x, t_eval, N;
             drift   = zeros(length(x)),     # constant drift vector μ
             sigma   = 1.0,                   # isotropic diffusion coefficient σ
             n_steps = 0,                      # 0 → exact (constant coeff only)
             drift_fn = nothing,               # μ(t, X) for non-constant drift
             antithetic = true,
             gpu  = HAS_GPU[],
             FT   = gpu ? Float32 : Float64)

Estimate `u(t_eval, x)` for the PDE  ∂ₜu = μ·∇u + ½σ²Δu,  u(0,·)=g
by Monte Carlo over `N` SDE paths.  Returns `FKResult`.

# Arguments
- `ic`:  initial condition `g(X)` where `X` is `(d, N)` → length-`N` vector
- `x`:   evaluation point (Vector{Float64}, length d)
- `t_eval`: evaluation time (> 0)
- `N`:   total number of paths (will be rounded to even if antithetic)
"""
function solve_mc(ic, x::Vector{Float64}, t_eval::Float64, N::Int;
                  drift::Vector{Float64} = zeros(length(x)),
                  sigma::Float64 = 1.0,
                  n_steps::Int = 0,
                  drift_fn = nothing,
                  antithetic::Bool = true,
                  gpu::Bool = HAS_GPU[],
                  FT::Type{<:AbstractFloat} = gpu ? Float32 : Float64)

    t0 = time()
    d  = length(x)
    Nh = antithetic ? N ÷ 2 : N

    # Reshape x for broadcasting: (d, 1)
    x_d  = to_dev(reshape(FT.(x), d, 1), Val(gpu))

    if n_steps == 0 && drift_fn === nothing
        # ── Exact solution (constant coefficients) ──────────────────────
        μ_d  = to_dev(reshape(FT.(drift), d, 1), Val(gpu))
        Z    = dev_randn(FT, d, Nh; gpu)
        sqt  = FT(sqrt(t_eval))
        σ_f  = FT(sigma)
        t_f  = FT(t_eval)

        X    = x_d .+ μ_d .* t_f .+ σ_f .* sqt .* Z
        vals = ic(X)

        if antithetic
            Xa    = x_d .+ μ_d .* t_f .- σ_f .* sqt .* Z
            valsa = ic(Xa)
            all_v = vcat(Array(vec(vals)), Array(vec(valsa)))
        else
            all_v = Array(vec(vals))
        end
    else
        # ── Euler-Maruyama ──────────────────────────────────────────────
        ns = n_steps > 0 ? n_steps : max(100, round(Int, t_eval * 100))
        dt = FT(t_eval / ns)
        sq = FT(sqrt(t_eval / ns))
        σ_f = FT(sigma)

        X  = repeat(x_d, 1, Nh)                     # (d, Nh)
        Xa = antithetic ? copy(X) : nothing

        for k in 0:ns-1
            tk = FT(k * Float64(dt))
            dW = dev_randn(FT, d, Nh; gpu) .* sq

            if drift_fn !== nothing
                μk = drift_fn(tk, X)                  # (d, Nh)
            else
                μk = to_dev(reshape(FT.(drift), d, 1), Val(gpu))
            end
            X .= X .+ μk .* dt .+ σ_f .* dW

            if antithetic
                if drift_fn !== nothing
                    μka = drift_fn(tk, Xa)
                else
                    μka = μk
                end
                Xa .= Xa .+ μka .* dt .- σ_f .* dW
            end
        end

        vals = ic(X)
        if antithetic
            valsa = ic(Xa)
            all_v = vcat(Array(vec(vals)), Array(vec(valsa)))
        else
            all_v = Array(vec(vals))
        end
    end

    μ_est = mean(all_v)
    σ_est = std(all_v) / sqrt(length(all_v))
    return FKResult(μ_est, σ_est, length(all_v), time() - t0)
end


# ---------------------------------------------------------------------------
#  solve_fk — full Feynman-Kac with potential V and source f
#
#  u(t,x) = 𝔼ₓ[ g(Xₜ) e^{−∫V} + ∫₀ᵗ f(t−s,Xₛ) e^{−∫₀ˢV} ds ]
# ---------------------------------------------------------------------------
"""
    solve_fk(ic, x, t_eval, N, n_steps;
             drift    = zeros(length(x)),
             drift_fn = nothing,
             sigma    = 1.0,
             potential  = nothing,    # V(t, X) → (N,)
             source     = nothing,    # f(t, X) → (N,)
             antithetic = true,
             gpu = HAS_GPU[], FT = ...)

Full Feynman-Kac Monte Carlo for PDEs with potential V and/or source f.
"""
function solve_fk(ic, x::Vector{Float64}, t_eval::Float64, N::Int, n_steps::Int;
                  drift::Vector{Float64} = zeros(length(x)),
                  drift_fn = nothing,
                  sigma::Float64 = 1.0,
                  potential = nothing,
                  source = nothing,
                  antithetic::Bool = true,
                  gpu::Bool = HAS_GPU[],
                  FT::Type{<:AbstractFloat} = gpu ? Float32 : Float64)

    t0 = time()
    d  = length(x)
    Nh = antithetic ? N ÷ 2 : N
    dt = FT(t_eval / n_steps)
    sq = FT(sqrt(t_eval / n_steps))
    σ_f = FT(sigma)

    x_d = to_dev(reshape(FT.(x), d, 1), Val(gpu))
    X   = repeat(x_d, 1, Nh)
    Xa  = antithetic ? copy(X) : nothing

    # Accumulators
    V_acc  = dev_zeros(FT, Nh; gpu)         # ∫₀ˢ V dr
    f_acc  = dev_zeros(FT, Nh; gpu)         # ∫₀ˢ f·e^{−∫V} ds
    Va_acc = antithetic ? dev_zeros(FT, Nh; gpu) : nothing
    fa_acc = antithetic ? dev_zeros(FT, Nh; gpu) : nothing

    has_V = potential !== nothing
    has_f = source    !== nothing

    for k in 0:n_steps-1
        tk   = FT(k * Float64(dt))
        t_pde = FT(t_eval) - tk                      # PDE time for source

        dW = dev_randn(FT, d, Nh; gpu) .* sq

        # ── accumulate source BEFORE stepping (left-endpoint quadrature) ──
        if has_f
            fv  = source(t_pde, X)                    # f at PDE-time
            f_acc .= f_acc .+ fv .* exp.(.-V_acc) .* dt
            if antithetic
                fva = source(t_pde, Xa)
                fa_acc .= fa_acc .+ fva .* exp.(.-Va_acc) .* dt
            end
        end

        # ── accumulate potential ──
        if has_V
            Vv  = potential(tk, X)
            V_acc .= V_acc .+ Vv .* dt
            if antithetic
                Vva = potential(tk, Xa)
                Va_acc .= Va_acc .+ Vva .* dt
            end
        end

        # ── SDE step ──
        if drift_fn !== nothing
            μk = drift_fn(tk, X)
        else
            μk = to_dev(reshape(FT.(drift), d, 1), Val(gpu))
        end
        X .= X .+ μk .* dt .+ σ_f .* dW

        if antithetic
            if drift_fn !== nothing
                μka = drift_fn(tk, Xa)
            else
                μka = μk
            end
            Xa .= Xa .+ μka .* dt .- σ_f .* dW
        end
    end

    # Terminal contribution
    terminal = ic(X) .* exp.(.-V_acc)
    result   = terminal .+ f_acc

    if antithetic
        terminal_a = ic(Xa) .* exp.(.-Va_acc)
        result_a   = terminal_a .+ fa_acc
        all_v = vcat(Array(vec(result)), Array(vec(result_a)))
    else
        all_v = Array(vec(result))
    end

    μ_est = mean(all_v)
    σ_est = std(all_v) / sqrt(length(all_v))
    return FKResult(μ_est, σ_est, length(all_v), time() - t0)
end


# ---------------------------------------------------------------------------
#  estimate_fpt — first-passage time via Langevin SDE
#
#  dXₛ = −(1/ξ)∇U(Xₛ)ds + √(2D) dWₛ,   X₀ = x₀
#
#  Estimates 𝔼[τ] where τ = inf{t : Xₜ ∈ B_target}
# ---------------------------------------------------------------------------
"""
    estimate_fpt(grad_U, x0, target_center, target_radius, N;
                 xi=1.0, D=1.0, dt=1e-3, max_steps=10^6,
                 gpu=HAS_GPU[], FT=...)

Estimate mean first-passage time from `x0` to the ball B(target_center, target_radius)
under Langevin dynamics with potential U (user provides ∇U).

`grad_U(X)` takes `(d, N)` matrix, returns `(d, N)` matrix of gradients.
"""
function estimate_fpt(grad_U, x0::Vector{Float64},
                      target_center::Vector{Float64}, target_radius::Float64,
                      N::Int;
                      xi::Float64 = 1.0, D::Float64 = 1.0,
                      dt::Float64 = 1e-3, max_steps::Int = 10^6,
                      gpu::Bool = HAS_GPU[],
                      FT::Type{<:AbstractFloat} = gpu ? Float32 : Float64)
    t0_clock = time()
    d  = length(x0)
    σ  = FT(sqrt(2D))
    sq = FT(sqrt(dt))
    dt_f = FT(dt)
    inv_xi = FT(1.0 / xi)
    r2 = FT(target_radius^2)

    x_d = to_dev(reshape(FT.(x0), d, 1), Val(gpu))
    c_d = to_dev(reshape(FT.(target_center), d, 1), Val(gpu))

    X      = repeat(x_d, 1, N)                       # (d, N) active paths
    times  = dev_zeros(FT, N; gpu)                    # hitting times
    active = dev_ones(FT, N; gpu)                     # 1.0 if still running

    for step in 1:max_steps
        n_active = sum(Array(active))
        n_active < 0.5 && break                       # all paths have hit

        dW = dev_randn(FT, d, N; gpu) .* sq
        gU = grad_U(X)                                # (d, N)

        X .= X .- inv_xi .* gU .* dt_f .+ σ .* dW .* active'

        # Check which paths hit the target
        diff = X .- c_d
        dist2 = vec(sum(diff .^ 2, dims=1))           # (N,)
        hit   = FT.(dist2 .< r2) .* active            # newly hit
        times .= times .+ active .* dt_f              # accumulate time for active
        active .= active .* (FT(1) .- hit)            # deactivate hit paths
    end

    fpt_vals = Array(times)
    still_active = Array(active)
    n_completed = count(x -> x < FT(0.5), still_active)

    if n_completed == 0
        @warn "No paths reached the target within max_steps=$(max_steps) steps"
        return FKResult(NaN, NaN, 0, time() - t0_clock)
    end

    # Only average over completed paths
    completed_times = fpt_vals[still_active .< FT(0.5)]
    μ_est = mean(completed_times)
    σ_est = std(completed_times) / sqrt(length(completed_times))

    pct = round(100 * n_completed / N, digits=1)
    @info "FPT: $(n_completed)/$(N) paths completed ($(pct)%)"

    return FKResult(μ_est, σ_est, n_completed, time() - t0_clock)
end
