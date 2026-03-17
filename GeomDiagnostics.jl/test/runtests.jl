using Test
using GeomDiagnostics

@testset "Test GeomDiagnostics.jl" begin

    include("test_covariance_field.jl")
    include("test_sampler_posdepcovrwm.jl")
    include("test_sampler_rwm.jl")
    include("test_distgen.jl")

end
