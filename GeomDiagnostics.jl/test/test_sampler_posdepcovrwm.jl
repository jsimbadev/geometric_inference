using Test
using Random
using LinearAlgebra
using AbstractMCMC
using LogDensityProblems

using GeomDiagnostics.CovarianceFields
using GeomDiagnostics.Samplers
import GeomDiagnostics.NGPCAJson: NGPCAUnitDO

struct ConstantLogDensity end
LogDensityProblems.logdensity(::ConstantLogDensity, x) = 0.0
LogDensityProblems.dimension(::ConstantLogDensity) = 2
LogDensityProblems.capabilities(::ConstantLogDensity) = LogDensityProblems.LogDensityOrder{0}()

struct DummyModel <: AbstractMCMC.AbstractModel end

m = [2, 2]
centers = [0.0 3.0; 0.0 3.0]
weights = [Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2)]
eigenvalues = [1.0 1.0; 2.0 2.0]
sigma_sqrs = [1.0, 1.0]
activities = [1.0, 1.0]
alphas = [0.0, 0.0]
epsilons = [1e-8, 1e-8]
metadata = NGPCAUnitDO(m, centers, weights, eigenvalues, sigma_sqrs, activities, alphas, epsilons)

@testset "PosDepCovRWM sampler behavior" begin
    cov_field = CovarianceFields.construct_ngpcacf_from_data(metadata.centers, CovarianceFields.UseBallTree())
    sampler = Samplers.HardPositionDependentRWMSampler(cov_field, metadata)

    @testset "construct_covariance is symmetric and positive definite" begin
        md_bad = NGPCAUnitDO(
            [2],
            reshape([0.0, 0.0], 2, 1),
            [Matrix{Float64}(I, 2, 2)],
            reshape([-1.0, 0.5], 2, 1),
            [1.0],
            [1.0],
            [0.0],
            [1e-6],
        )
        cov = Samplers.construct_covariance(1, md_bad)
        @test cov ≈ transpose(cov)
        @test isposdef(Symmetric(cov))
    end

    @testset "propose returns state, covariance and logproposal" begin
        rng = MersenneTwister(1234)
        state = [0.2, -0.3]
        proposal = Samplers.propose(rng, state, sampler)

        @test proposal isa NamedTuple
        @test keys(proposal) == (:state, :covariance, :logproposal)
        @test length(proposal.state) == length(state)
        @test size(proposal.covariance) == (length(state), length(state))
        @test proposal.logproposal isa Real
        @test isposdef(Symmetric(proposal.covariance))
    end

    @testset "step accepts always when log acceptance is zero" begin
        model = AbstractMCMC.LogDensityModel(ConstantLogDensity())
        state = [0.1, -0.2]

        rng_for_expected = MersenneTwister(7)
        rng_for_step = deepcopy(rng_for_expected)

        expected = Samplers.propose(rng_for_expected, state, sampler).state
        next_state = AbstractMCMC.step(rng_for_step, model, sampler, state)

        @test next_state == expected
    end

    @testset "step errors for non-LogDensityModel targets" begin
        @test_throws ErrorException AbstractMCMC.step(MersenneTwister(1), DummyModel(), sampler, [0.0, 0.0])
    end
end
