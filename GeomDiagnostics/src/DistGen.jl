module DistGen


import Comonicon
using Random, Distributions

# """
# The purpose of this module is to provide a command line interface for the distributions functionality in GeomDiagnostics. It allows users to easily generate and visualize distributions of geometric properties from their data.
# """

# Supported distributions
const SUPPORTED_DISTRIBUTIONS = Dict(
    "normal" => Normal(0, 1)
)

function generate_point_cloud(distribution::String, num_points::Int, dim::Int=2, seed::Int=42)
    if ! (distribution in keys(SUPPORTED_DISTRIBUTIONS))
        @error "Distribution is not supported" distribution=distribution
    end

    @info "SEED: " seed=seed 
    distribution_object = SUPPORTED_DISTRIBUTIONS[distribution]

    rng = MersenneTwister(seed)
    point_cloud = rand(rng, distribution_object, num_points, dim)

    println(point_cloud)

end 

end