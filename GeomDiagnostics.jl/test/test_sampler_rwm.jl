using Test
using Random
using LinearAlgebra
using AbstractMCMC
using LogDensityProblems

using GeomDiagnostics.Samplers

struct ConstantLogDensity end
LogDensityProblems.logdensity(::ConstantLogDensity, x) = 0.0
LogDensityProblems.dimension(::ConstantLogDensity) = 2
LogDensityProblems.capabilities(::ConstantLogDensity) = LogDensityProblems.LogDensityOrder{0}()

struct DummyModel <: AbstractMCMC.AbstractModel end


cov_field = reshape([1.0,0.0,0.0,1.0], 2,2)
sampler = Samplers.RWMSampler(cov_field)

@testset "RWM sampler behavior" begin
    ϵ = 0.5
    dummy_state = [1.0,2.0]
    @test Samplers.dimension(sampler) == 2
    @test Samplers.local_covariance(dummy_state, sampler) == Samplers.local_covariance(dummy_state .+ ϵ, sampler)

end
