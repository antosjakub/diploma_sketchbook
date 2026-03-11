using DelimitedFiles
using JSON3

# User-editable parameters
function my_func(x::Vector{Float64}, t::Float64)
    return sum(sin.(x .* π)) * exp(-t)  # Your function here
end

d = 5  # Space dimension
t_fixed = 1.0
varying_dims = [1, 2]  # 1-based indices to vary
fixed_vals = [0.5, 1.2, -0.3]  # Fixed values for remaining d-2 dims
ranges = [range(-π, π, length=100), range(-π, π, length=100)]  # Varying ranges
output_prefix = "slice_data"  # Base name for files
output_prefix_metadata = "slice_data"

# Validate inputs
@assert length(fixed_vals) == d - 2 "fixed_vals must have length d-2"
@assert length(varying_dims) == 2 "varying_dims must have exactly 2 elements"
@assert all(1 <= dim <= d for dim in varying_dims) "varying_dims out of 1:d"
fixed_dims = setdiff(1:d, varying_dims)
fixed_part = Dict(zip(fixed_dims, fixed_vals))

# Evaluate grid (Z[row, col] where row ~ x2, col ~ x1)
Z_flat = Float64[]
x1_grid = ranges[1]
x2_grid = ranges[2]
for x2 in x2_grid, x1 in x1_grid  # Note: outer x2 for row-major
    x = zeros(d)
    x[varying_dims[1]] = x1
    x[varying_dims[2]] = x2
    for (dim, val) in fixed_part
        x[dim] = val
    end
    push!(Z_flat, my_func(x, t_fixed))
end
Z = reshape(Z_flat, length(x2_grid), length(x1_grid))

# Save Z as space-delimited text (universal, load with np.loadtxt)
writedlm("$output_prefix.txt", Z, ' ', header=false)

# Save metadata JSON
metadata = Dict(
    "d" => d,
    "t_fixed" => t_fixed,
    "varying_dims" => varying_dims,
    "fixed_dims" => collect(fixed_dims),
    "fixed_vals" => fixed_vals,
    "x1_range" => [minimum(x1_grid), maximum(x1_grid), length(x1_grid)],
    "x2_range" => [minimum(x2_grid), maximum(x2_grid), length(x2_grid)],
    "Z_shape" => size(Z),
    "Z_min" => minimum(Z),
    "Z_max" => maximum(Z)
)
open("$output_prefix_metadata.json", "w") do f
    JSON3.pretty(f, JSON3.write(metadata))
end

println("Saved: $output_prefix.txt (Z matrix) and $output_prefix_metadata.json")
