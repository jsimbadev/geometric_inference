function generate_samples(distribution::String, num_points::Int, seed::Int=42)
    @info "SEED: " seed=seed
    distribution_object = get_sample_source(distribution)
    rng = Random.MersenneTwister(seed)
    sample(rng, distribution_object, num_points)
end

function generate_point_cloud(distribution::String, num_points::Int, seed::Int=42, plot_points::Bool=false)
    point_cloud = generate_samples(distribution, num_points, seed)

    maybe_plot_point_cloud(point_cloud, distribution, num_points, seed, plot_points)

    Io.write_samples(
        transpose(point_cloud),
        joinpath("$(generate_outputs_dir())", "$(distribution)_samples.csv"),
    )
end
