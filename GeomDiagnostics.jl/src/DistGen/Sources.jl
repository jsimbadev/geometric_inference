using AbstractMCMC

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

register_sample_source!(
    "normal",
    MvnNormalSampleSource(Distributions.MvNormal([0, 0], LinearAlgebra.Matrix(LinearAlgebra.I, 2, 2))),
)
register_sample_source!(
    "banana",
    BananaSampleSource(1, 1, Distributions.MvNormal([0, 0], reshape([1, 0.9, 0.9, 1], 2, 2))),
)
