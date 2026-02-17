module DistGen


import Comonicon
using Random, Distributions, LinearAlgebra, Plots

# """
# The purpose of this module is to provide a command line interface for the distributions functionality in GeomDiagnostics. It allows users to easily generate and visualize distributions of geometric properties from their data.
# """

# Supported distributions
const SUPPORTED_DISTRIBUTIONS = Dict(
    "normal" => MvNormal([0,0], Matrix(I, 2, 2))
)

const MANUALLY_CHECK_FIT = true

function generate_file_name(distribution::String, num_points::Int, seed::Int)
    project_root = normpath(joinpath(@__DIR__, ".."))
    output_dir = joinpath(project_root, "outputs", "plots")
    mkpath(output_dir)
    timestamp = string(round(Int, time()))
    output_path = joinpath(
        output_dir,
        "point_cloud_$(distribution)_n$(num_points)_seed$(seed)_$(timestamp).png",
    )
end


function generate_point_cloud(distribution::String, num_points::Int, seed::Int=42, plot_points::Bool=false)
    if ! (distribution in keys(SUPPORTED_DISTRIBUTIONS))
        @error "Distribution is not supported" distribution=distribution
    end

    @info "SEED: " seed=seed 
    distribution_object = SUPPORTED_DISTRIBUTIONS[distribution]

    rng = MersenneTwister(seed)
    point_cloud = rand(rng, distribution_object, num_points)


    # println(point_cloud)
    if plot_points
        @info "Plotting the cloud points"
        plt = scatter(
            point_cloud[1, :],
            point_cloud[2, :];
            xlabel="x",
            ylabel="y",
            title="Point cloud ($(distribution), n=$(num_points), seed=$(seed))",
            label=false,
        )
        output_path = generate_file_name(distribution, num_points, seed)
        savefig(plt, output_path)
        @info "Saved point cloud plot" output_path=output_path
    end

    if MANUALLY_CHECK_FIT
        @info "Got to this line"
        params = fit(MvNormal, point_cloud)
        @info "The estimated parameters for the $(distribution): $(params)"
    end

end 

end
