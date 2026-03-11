#using Printf, Random
#include("src/core.jl")
#include("src/mlmc.jl")
#include("src/FeynmanKac.jl")
using Pkg; Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "FeynmanKac.jl"))
using .FeynmanKac
using Printf, Random


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
function run_travelling_gaussian(; d::Int=10, x0,t0, N::Int=10^6, n_steps::Int=200)
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

    x = fill(x0, d)
    t_eval = t0

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

    ## Also try MLMC
    #println("  Running MLMC...")
    #res_mlmc = solve_mlmc(ic, x, t_eval;
    #                      drift=μ_sde, sigma=σ_sde,
    #                      potential=potential, source=source_fn,
    #                      target_rmse=1e-3)
    #@printf("  MLMC:      u = %+.8f  (exact = %+.8f,  err = %.2e,  σ = %.2e,  %.3fs)\n",
    #        res_mlmc.value, exact, abs(res_mlmc.value - exact), res_mlmc.std_error, res_mlmc.elapsed)

    ## ── Importance sampling via Girsanov ──────────────────────────────────
    ## Optimal IS drift: θ* = σ ∇log u.  Since u is known analytically:
    ##   log u = −α Σ(aᵢxᵢ − bᵢ + cᵢτ)² − βτ + log cos(γτ)
    ##   ∇ₓ log u = −2α aᵢ(aᵢxᵢ − bᵢ + cᵢτ)   (component i)
    ##   θ* = σ · ∇log u
    ## where τ = t_eval − t_sde is the PDE time.
    #println("  Running EM MC + importance sampling...")
    #function is_drift_fn(t_sde, X)
    #    FT = eltype(X)
    #    a_d = to_dev_like(FT.(a_vec), X)
    #    b_d = to_dev_like(FT.(b_vec), X)
    #    c_d = to_dev_like(FT.(c_vec), X)
    #    τ = FT(t_eval) - t_sde                         # PDE time
    #    z = a_d .* X .- b_d .+ c_d .* τ
    #    return FT(σ_sde) .* (FT(-2α) .* a_d .* z)      # σ · ∇log u
    #end

    #res_is = solve_fk(ic, x, t_eval, N, n_steps;
    #                  drift=μ_sde, sigma=σ_sde,
    #                  potential=potential, source=source_fn,
    #                  is_drift=is_drift_fn)
    #@printf("  EM MC+IS:  u = %+.8f  (exact = %+.8f,  err = %.2e,  σ = %.2e,  %.3fs)\n",
    #        res_is.value, exact, abs(res_is.value - exact), res_is.std_error, res_is.elapsed)
    #@printf("  Variance reduction factor: %.1fx\n", res.std_error / max(res_is.std_error, 1e-30))

    return res
end



function main()

    result = run_travelling_gaussian(d=4, x0=0.5, t0=0.3, n_steps=200, N=1000)
    println(result.value)

end

main()
