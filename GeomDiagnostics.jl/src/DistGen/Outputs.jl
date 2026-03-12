function generate_outputs_dir()
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    output_dir = joinpath(project_root, "outputs")
    output_dir
end

function generate_file_name(distribution::String, num_points::Int, seed::Int)
    output_dir = joinpath(generate_outputs_dir(), "plots")
    mkpath(output_dir)
    timestamp = string(round(Int, time()))
    joinpath(
        output_dir,
        "point_cloud_$(distribution)_n$(num_points)_seed$(seed)_$(timestamp).png",
    )
end

function maybe_plot_point_cloud(
    point_cloud::AbstractMatrix,
    distribution::String,
    num_points::Int,
    seed::Int,
    plot_points::Bool,
)
    if !plot_points
        return nothing
    end

    if size(point_cloud, 1) < 2
        error("Plotting requires at least 2 dimensions")
    end

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
