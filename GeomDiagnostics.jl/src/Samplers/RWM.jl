using Random, LinearAlgebra

struct RWMSampler{Cov<:Matrix} <: AbstractPositionDependentRWMSampler
    Σ::Cov
end

function local_covariance(::AbstractVector, sampler::RWMSampler)
    sampler.Σ
end

function dimension(sampler::RWMSampler)
    size(sampler.Σ)[begin]
end



