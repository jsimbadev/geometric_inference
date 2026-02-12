module GeomDiagnostics

include("DistGen.jl")

using Comonicon

Comonicon.@main function main(distribution::String, num_points::Int; dim::Int=2, seed::Int=42)
    DistGen.generate_point_cloud(distribution, num_points, dim, seed)
end

end