module GeomDiagnostics

include("io.jl")
include("NGPCAJson.jl")
include("CovarianceFields.jl")
include("Samplers.jl")
include("DistGen.jl")

using Comonicon

Comonicon.@main function main(distribution_or_config::String, num_points::Int; seed::Int=42, plot_points::Bool=false)
    if endswith(lowercase(distribution_or_config), ".json")
        DistGen.generate_point_cloud_from_config(distribution_or_config, num_points)
        return
    end
    DistGen.generate_point_cloud(distribution_or_config, num_points, seed, plot_points)
end

end
