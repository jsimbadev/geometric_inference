using Test
using Random
using AbstractMCMC
using LogDensityProblems

using GeomDiagnostics.CovarianceFields
using GeomDiagnostics.DistGen
using GeomDiagnostics.Samplers
import GeomDiagnostics.NGPCAJson: NGPCAUnitDO

struct ConstantLogDensityDG end
LogDensityProblems.logdensity(::ConstantLogDensityDG, x) = 0.0
LogDensityProblems.dimension(::ConstantLogDensityDG) = 2
LogDensityProblems.capabilities(::ConstantLogDensityDG) = LogDensityProblems.LogDensityOrder{0}()

function distgen_test_sampler()
    m = [2, 2]
    centers = [0.0 3.0; 0.0 3.0]
    weights = [Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2)]
    eigenvalues = [1.0 1.0; 2.0 2.0]
    sigma_sqrs = [1.0, 1.0]
    activities = [1.0, 1.0]
    alphas = [0.0, 0.0]
    epsilons = [1e-8, 1e-8]
    metadata = NGPCAUnitDO(m, centers, weights, eigenvalues, sigma_sqrs, activities, alphas, epsilons)
    cov_field = CovarianceFields.construct_ngpcacf_from_data(metadata.centers, CovarianceFields.UseBallTree())
    Samplers.HardPositionDependentRWMSampler(cov_field, metadata)
end

@testset "DistGen gateway behavior" begin
    @test "normal" in DistGen.supported_distributions()
    @test "banana" in DistGen.supported_distributions()

    @testset "generate_samples works for direct sources" begin
        normal_points = DistGen.generate_samples("normal", 5, 7)
        banana_points = DistGen.generate_samples("banana", 5, 7)

        @test size(normal_points) == (2, 5)
        @test size(banana_points) == (2, 5)
    end

    @testset "MCMC sample source integrates with gateway" begin
        sampler = distgen_test_sampler()
        model = AbstractMCMC.LogDensityModel(ConstantLogDensityDG())
        mcmc_source = DistGen.MCMCSampleSource(model, sampler, [0.0, 0.0], 5, 2)
        source_name = "test_const_mcmc"

        DistGen.register_sample_source!(source_name, mcmc_source)
        points = DistGen.generate_samples(source_name, 4, 11)

        @test size(points) == (2, 4)
        @test points[:, 1] != points[:, 2]

        delete!(DistGen.SUPPORTED_DISTRIBUTIONS, source_name)
    end
end
