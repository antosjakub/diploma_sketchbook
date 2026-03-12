using DelimitedFiles
using JSON3



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
d = 6
δ = 1.0                                          # diffusion (δ=1.0 in PyTorch)
α = 7.4
β = 0.2
γ = 1.9*pi
a_vec =  fill(0.9, d) + 0.2*rand(d)
b_vec =  fill(0.4, d) + 0.2*rand(d)
c_vec = fill(-0.3, d) + 0.6*rand(d)
w = -2δ * α * sum(a_vec .^ 2)                   # w < 0 !
function run_travelling_gaussian(; d::Int=10, x::Vector{Float64}, t_eval::Float64, N::Int=10^6, n_steps::Int=200)

    println("\n" * "="^70)
    println("  Problem 3: Travelling Gaussian  (d=$d)")
    println("="^70)

    # Parameters (matching the PyTorch TravellingGaussPacket_v2)

    # SDE coefficients
    v_drift = -c_vec ./ a_vec                        # vᵢ = −cᵢ/aᵢ
    μ_sde   = -v_drift                               # SDE drift = −v
    σ_sde   = sqrt(2δ)

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
    #println("  Running MLMC...")
    #res_mlmc = solve_mlmc(ic, x, t_eval;
    #                      drift=μ_sde, sigma=σ_sde,
    #                      potential=potential, source=source_fn,
    #                      target_rmse=1e-2)
    #@printf("  MLMC:      u = %+.8f  (exact = %+.8f,  err = %.2e,  σ = %.2e,  %.3fs)\n",
    #        res_mlmc.value, exact, abs(res_mlmc.value - exact), res_mlmc.std_error, res_mlmc.elapsed)

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

    return res.value, res_is.value, exact
end



function my_func(x::Vector{Float64}, t::Float64)
    d = length(x)
    fk, fk_mlmlc, ex = run_travelling_gaussian(d=d, x=x, t_eval=t, n_steps=100, N=10^6)
    return fk, fk_mlmlc, ex
end


#d = 4
#x = fill(0.5, d)
#t_eval = 0.3



t_fixed = 0.1
nx = 5
varying_dims = [1, 2]  # 1-based indices to vary
#fixed_vals = [0.5, 0.6, 0.3]  # Fixed values for remaining d-2 dims
#fixed_vals = [0.3, 0.4]  # Fixed values for remaining d-2 dims
fixed_vals =  fill(0.25, d-2) + 0.5*rand(d-2)
ranges = [range(0, 1, length=nx), range(0, 1, length=nx)]  # Varying ranges
output_prefix = "slice_data"  # Base name for files
output_prefix_metadata = "slice_data"

# Validate inputs
@assert length(fixed_vals) == d - 2 "fixed_vals must have length d-2"
@assert length(varying_dims) == 2 "varying_dims must have exactly 2 elements"
@assert all(1 <= dim <= d for dim in varying_dims) "varying_dims out of 1:d"
fixed_dims = setdiff(1:d, varying_dims)
fixed_part = Dict(zip(fixed_dims, fixed_vals))

# Evaluate grid (Z[row, col] where row ~ x2, col ~ x1)
Z_flat_fk = Float64[]
Z_flat_fk_mlmc = Float64[]
Z_flat_ex = Float64[]
x1_grid = ranges[1]
x2_grid = ranges[2]
for x2 in x2_grid, x1 in x1_grid  # Note: outer x2 for row-major
    x = zeros(d)
    x[varying_dims[1]] = x1
    x[varying_dims[2]] = x2
    for (dim, val) in fixed_part
        x[dim] = val
    end
    println(x2, ", ", x1)
    fk, fk_mlmlc, ex = my_func(x, t_fixed)
    push!(Z_flat_fk, fk)
    push!(Z_flat_fk_mlmc, fk)
    push!(Z_flat_ex, ex)
end
Z_fk = reshape(Z_flat_fk, length(x2_grid), length(x1_grid))
Z_flat_fk_mlmc = reshape(Z_flat_fk_mlmc, length(x2_grid), length(x1_grid))
Z_ex = reshape(Z_flat_ex, length(x2_grid), length(x1_grid))

# Save Z as space-delimited text (universal, load with np.loadtxt)
writedlm("$(output_prefix)_fk.txt", Z_fk, ' ', header=false)
writedlm("$(output_prefix)_fk_mlmc.txt", Z_flat_fk_mlmc, ' ', header=false)
writedlm("$(output_prefix)_ex.txt", Z_ex, ' ', header=false)

# Save metadata JSON
metadata = Dict(
    "d" => d,
    "t_fixed" => t_fixed,
    "varying_dims" => varying_dims,
    "fixed_dims" => collect(fixed_dims),
    "fixed_vals" => fixed_vals,
    "x1_range" => [minimum(x1_grid), maximum(x1_grid), length(x1_grid)],
    "x2_range" => [minimum(x2_grid), maximum(x2_grid), length(x2_grid)],
    "Z_shape" => size(Z_ex),
    "Z_min" => minimum(Z_ex),
    "Z_max" => maximum(Z_ex)
)
open("$output_prefix_metadata.json", "w") do f
    JSON3.pretty(f, JSON3.write(metadata))
end

println("Saved: $output_prefix.txt (Z matrix) and $output_prefix_metadata.json")
