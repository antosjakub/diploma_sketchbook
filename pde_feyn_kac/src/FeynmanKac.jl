module FeynmanKac

using CUDA, Statistics, Printf, Random

export FKResult, solve_mc, solve_fk, solve_mlmc, estimate_fpt, to_dev_like

# ---------------------------------------------------------------------------
#  Result type
# ---------------------------------------------------------------------------
struct FKResult
    value::Float64
    std_error::Float64
    n_samples::Int
    elapsed::Float64
end

function Base.show(io::IO, r::FKResult)
    @printf(io, "FKResult(value=%.8e, std_err=%.2e, N=%d, time=%.3fs)", r.value, r.std_error, r.n_samples, r.elapsed)
end

# ---------------------------------------------------------------------------
#  Device helpers — transparently use GPU or CPU
# ---------------------------------------------------------------------------
const HAS_GPU = Ref(false)

function __init__()
    HAS_GPU[] = CUDA.functional()
    if HAS_GPU[]
        @info "FeynmanKac: CUDA GPU detected — using GPU acceleration"
    else
        @info "FeynmanKac: No GPU detected — falling back to CPU (multithreaded)"
    end
end

to_dev(x::AbstractArray, ::Val{true})  = CuArray(x)
to_dev(x::AbstractArray, ::Val{false}) = Array(x)
dev_randn(::Type{T}, dims...; gpu=true) where T = gpu ? CUDA.randn(T, dims...) : randn(T, dims...)
dev_zeros(::Type{T}, dims...; gpu=true) where T = gpu ? CUDA.zeros(T, dims...) : zeros(T, dims...)
dev_ones(::Type{T}, dims...; gpu=true)  where T = gpu ? CUDA.ones(T, dims...)  : ones(T, dims...)

"""
    to_dev_like(src, reference)

Convert `src` to live on the same device as `reference` (GPU ↔ CPU).
Use this in user-supplied closures to match captured vectors to the
solver's array type without importing CUDA directly.

# Example
    ic(X) = begin
        a_d = to_dev_like(Float32.(a_vec), X)   # moves a_vec to wherever X lives
        vec(prod(sin.(a_d .* X), dims=1))
    end
"""
to_dev_like(src::AbstractArray, ref::AbstractArray) = src
to_dev_like(src::AbstractArray, ref::CuArray)       = CuArray(src)
to_dev_like(src::CuArray,       ref::AbstractArray) = Array(src)
to_dev_like(src::CuArray,       ref::CuArray)       = src

include("core.jl")
include("mlmc.jl")

end # module
