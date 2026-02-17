module GeomDiagnostics

include("DistGen.jl")

using Comonicon

Comonicon.@main function main(distribution::String, num_points::Int; seed::Int=42, plot_points::Bool=false)
    DistGen.generate_point_cloud(distribution, num_points, seed, plot_points)
end

end