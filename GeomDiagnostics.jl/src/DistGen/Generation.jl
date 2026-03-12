function generate_samples(distribution::String, num_points::Int, seed::Int=42)
    @info "SEED: " seed=seed
    distribution_object = get_sample_source(distribution)
    rng = Random.MersenneTwister(seed)
    sample(rng, distribution_object, num_points)
end

function generate_samples(config::DistGenConfig, num_points::Int)
    @info "SEED: " seed=config.Seed
    distribution_object = build_sample_source(config)
    rng = Random.MersenneTwister(config.Seed)
    sample(rng, distribution_object, num_points)
end

function generate_samples_from_config(config_path::AbstractString, num_points::Int)
    config = read_distgen_config(config_path)
    generate_samples(config, num_points)
end

function generate_point_cloud(distribution::String, num_points::Int, seed::Int=42, plot_points::Bool=false)
    point_cloud = generate_samples(distribution, num_points, seed)

    maybe_plot_point_cloud(point_cloud, distribution, num_points, seed, plot_points)

    Io.write_samples(
        transpose(point_cloud),
        joinpath("$(generate_outputs_dir())", "$(distribution)_samples.csv"),
    )
end

function generate_point_cloud(config::DistGenConfig, num_points::Int)
    point_cloud = generate_samples(config, num_points)

    maybe_plot_point_cloud(point_cloud, config.Source, num_points, config.Seed, config.PlotPoints)

    Io.write_samples(
        transpose(point_cloud),
        joinpath("$(generate_outputs_dir())", "$(config.Source)_samples.csv"),
    )
end

function generate_point_cloud_from_config(config_path::AbstractString, num_points::Int)
    config = read_distgen_config(config_path)
    generate_point_cloud(config, num_points)
end
