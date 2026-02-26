module DistGen


import Comonicon
using ..Io
using Random, Distributions, LinearAlgebra, Plots

# """
# The purpose of this module is to provide a command line interface for the distributions functionality in GeomDiagnostics. It allows users to easily generate and visualize distributions of geometric properties from their data.
# """

abstract type AbstractSampleSource end

struct MvnNormalSampleSource{MvnNormalD<:Distributions.AbstractMvNormal} <: AbstractSampleSource
    Mvn::MvnNormalD
end

# https://lutpub.lut.fi/bitstream/handle/10024/36631/isbn9789516976627.pdf?sequence=1&isAllowed=y
# https://cran.r-project.org/web/packages/FME/vignettes/FMEmcmc.pdf
struct BananaSampleSource{T<:Real, MvnNormalD<:Distributions.AbstractMvNormal} <: AbstractSampleSource
    a::T
    b::T
    Mvn::MvnNormalD
end

function _generate_mvn_gaussian_coordinates(rng::Random.AbstractRNG, d::Distributions.AbstractMvNormal, num::Int=1)
    return rand(rng, d, num)
end     


function banana_transform(gaussian_points::AbstractMatrix, a::Real, b::Real)
    d, n = size(gaussian_points)
    if d < 2
        error("Banana transform requires at least 2 dimensions. Got d=$d")
    end

    transformed = copy(gaussian_points)

    # 2D banana warp on the first two coordinates:
    # x1' = x1
    # x2' = x2 + b * (x1^2 - a^2)
    # For d > 2, coordinates 3..d are left unchanged.
    @inbounds for j in 1:n
        x1 = gaussian_points[1, j]
        transformed[2, j] = gaussian_points[2, j] + b * (x1^2 - a^2)
    end

    return transformed
end

function sample(rng::Random.AbstractRNG, ss::MvnNormalSampleSource, num::Int=1)
    return rand(rng, ss.Mvn, num) 
end

function sample(rng::Random.AbstractRNG, ss::BananaSampleSource, num::Int=1)
    gaussian_coords = _generate_mvn_gaussian_coordinates(rng, ss.Mvn, num)
    # Transform the Gaussian coordinates
    return banana_transform(gaussian_coords, ss.a, ss.b)

end


# Supported distributions
const SUPPORTED_DISTRIBUTIONS = Dict{String, AbstractSampleSource}(
    "normal" => MvnNormalSampleSource(Distributions.MvNormal([0, 0], LinearAlgebra.Matrix(LinearAlgebra.I, 2, 2))),
    "banana" => BananaSampleSource(1, 1, Distributions.MvNormal([0, 0], reshape([1, 0.9, 0.9, 1], 2,2 )))
)

const MANUALLY_CHECK_FIT = true


function generate_outputs_dir()
    project_root = normpath(joinpath(@__DIR__, ".."))
    output_dir = joinpath(project_root, "outputs")
    return output_dir
end

function generate_file_name(distribution::String, num_points::Int, seed::Int)
    output_dir = joinpath(generate_outputs_dir(), "plots")
    mkpath(output_dir)
    timestamp = string(round(Int, time()))
    output_path = joinpath(
        output_dir,
        "point_cloud_$(distribution)_n$(num_points)_seed$(seed)_$(timestamp).png",
    )
    return output_path
end


function generate_point_cloud(distribution::String, num_points::Int, seed::Int=42, plot_points::Bool=false)
    if !haskey(SUPPORTED_DISTRIBUTIONS, distribution)
        @error "Distribution is not supported" distribution=distribution
    end

    @info "SEED: " seed=seed 
    distribution_object = SUPPORTED_DISTRIBUTIONS[distribution]

    rng = Random.MersenneTwister(seed)
    point_cloud = sample(rng, distribution_object, num_points)


    # println(point_cloud)
    if plot_points
        @info "Plotting the cloud points"
        plt = Plots.scatter(
            point_cloud[1, :],
            point_cloud[2, :];
            xlabel="x",
            ylabel="y",
            title="Point cloud ($(distribution), n=$(num_points), seed=$(seed))",
            label=false,
        )
        output_path = generate_file_name(distribution, num_points, seed)
        Plots.savefig(plt, output_path)
        @info "Saved point cloud plot" output_path=output_path
    end

    # TODO is this useful and worth keeping
    # if MANUALLY_CHECK_FIT
    #     @info "Got to this line"
    #     params = fit(MvNormal, point_cloud)
    #     @info "The estimated parameters for the $(distribution): $(params)"
    # end


    Io.write_samples(
        transpose(point_cloud),
        joinpath("$(generate_outputs_dir())", "$(distribution)_samples.csv")
    )


end 

end
