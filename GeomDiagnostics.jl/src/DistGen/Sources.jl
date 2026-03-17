using AbstractMCMC
using LogDensityProblems
using ..CovarianceFields
using ..Samplers
import ..NGPCAJson: NGPCAUnitDO

abstract type AbstractSampleSource end

struct MvnNormalSampleSource{MvnNormalD<:Distributions.AbstractMvNormal} <: AbstractSampleSource
    Mvn::MvnNormalD
end

# https://lutpub.lut.fi/bitstream/handle/10024/36631/isbn9789516976627.pdf?sequence=1&isAllowed=y
# https://cran.r-project.org/web/packages/FME/vignettes/FMEmcmc.pdf
struct BananaSampleSource{T<:Real, MvnNormalD<:Distributions.AbstractMvNormal} <: AbstractSampleSource
    a::T
    b::T
    Mvn::MvnNormalD
end

struct MCMCSampleSource{
    M<:AbstractMCMC.AbstractModel,
    S<:AbstractMCMC.AbstractSampler,
    V<:AbstractVector,
} <: AbstractSampleSource
    Model::M
    Sampler::S
    InitialState::V
    BurnIn::Int
    Thinning::Int
end

struct MvNormalLogDensity{D<:Distributions.AbstractMvNormal}
    Dist::D
end

struct BananaLogDensity{T<:Real, D<:Distributions.AbstractMvNormal}
    a::T
    b::T
    Base::D
end

LogDensityProblems.logdensity(ld::MvNormalLogDensity, x) = logpdf(ld.Dist, x)
LogDensityProblems.dimension(ld::MvNormalLogDensity) = length(mean(ld.Dist))
LogDensityProblems.capabilities(::MvNormalLogDensity) = LogDensityProblems.LogDensityOrder{0}()

LogDensityProblems.dimension(ld::BananaLogDensity) = length(mean(ld.Base))
LogDensityProblems.capabilities(::BananaLogDensity) = LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(ld::BananaLogDensity, x)
    length(x) < 2 && error("Banana log density requires at least 2 dimensions")
    z = copy(x)
    z[2] = x[2] - ld.b * (x[1]^2 - ld.a^2)
    logpdf(ld.Base, z)
end

function _generate_mvn_gaussian_coordinates(rng::Random.AbstractRNG, d::Distributions.AbstractMvNormal, num::Int=1)
    rand(rng, d, num)
end

function banana_transform(gaussian_points::AbstractMatrix, a::Real, b::Real)
    d, n = size(gaussian_points)
    if d < 2
        error("Banana transform requires at least 2 dimensions. Got d=$d")
    end

    transformed = copy(gaussian_points)

    @inbounds for j in 1:n
        x1 = gaussian_points[1, j]
        transformed[2, j] = gaussian_points[2, j] + b * (x1^2 - a^2)
    end

    transformed
end

function sample(rng::Random.AbstractRNG, ss::MvnNormalSampleSource, num::Int=1)
    rand(rng, ss.Mvn, num)
end

function sample(rng::Random.AbstractRNG, ss::BananaSampleSource, num::Int=1)
    gaussian_coords = _generate_mvn_gaussian_coordinates(rng, ss.Mvn, num)
    banana_transform(gaussian_coords, ss.a, ss.b)
end

function sample(rng::Random.AbstractRNG, ss::MCMCSampleSource, num::Int=1)
    if num < 0
        error("num must be non-negative, got $num")
    end
    if ss.BurnIn < 0
        error("BurnIn must be non-negative, got $(ss.BurnIn)")
    end
    if ss.Thinning < 1
        error("Thinning must be at least 1, got $(ss.Thinning)")
    end

    state = copy(ss.InitialState)
    d = length(state)
    output = Matrix{eltype(state)}(undef, d, num)

    if num == 0
        return output
    end

    total_steps = ss.BurnIn + num * ss.Thinning
    collected = 0

    for i in 1:total_steps
        state = AbstractMCMC.step(rng, ss.Model, ss.Sampler, state)
        if i > ss.BurnIn && ((i - ss.BurnIn) % ss.Thinning == 0)
            collected += 1
            output[:, collected] = state
        end
    end

    output
end

const SAMPLE_SOURCE_REGISTRY = Dict{String, AbstractSampleSource}()
const SUPPORTED_DISTRIBUTIONS = SAMPLE_SOURCE_REGISTRY

function register_sample_source!(name::AbstractString, source::AbstractSampleSource)
    SAMPLE_SOURCE_REGISTRY[String(name)] = source
end

function get_sample_source(name::AbstractString)
    if !haskey(SAMPLE_SOURCE_REGISTRY, name)
        error("Distribution is not supported: $name")
    end
    SAMPLE_SOURCE_REGISTRY[name]
end

function supported_distributions()
    collect(keys(SAMPLE_SOURCE_REGISTRY))
end

# TODO Major - Generalize all of this for higher dimension experiments
function make_pdrwm_sampler_2d(; proposal_variance::Real=0.5)
    m = [2, 2]
    centers = [-2.0 2.0; -2.0 2.0]
    weights = [Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2)]
    eigenvalues = [proposal_variance proposal_variance; proposal_variance proposal_variance]
    sigma_sqrs = [1.0, 1.0]
    activities = [1.0, 1.0]
    alphas = [0.0, 0.0]
    epsilons = [1e-8, 1e-8]
    metadata = NGPCAUnitDO(m, centers, weights, eigenvalues, sigma_sqrs, activities, alphas, epsilons)
    cov_field = CovarianceFields.construct_ngpcacf_from_data(metadata.centers, CovarianceFields.UseBallTree())
    Samplers.HardPositionDependentRWMSampler(cov_field, metadata)
end

function make_rwm_sampler_2d(; proposal_variance::Real=0.5, dimension::Int=2)
    cov_field = proposal_variance * Matrix(I, dimension, dimension)
    Samplers.RWMSampler(cov_field)
end

register_sample_source!(
    "normal",
    MvnNormalSampleSource(Distributions.MvNormal([0, 0], LinearAlgebra.Matrix(LinearAlgebra.I, 2, 2))),
)
register_sample_source!(
    "banana",
    BananaSampleSource(1, 1, Distributions.MvNormal([0, 0], reshape([1, 0.9, 0.9, 1], 2, 2))),
)
