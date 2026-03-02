
using AbstractMCMC
using ..NGPCAJson
using ..CovarianceFields
using Distributions, Random, LinearAlgebra

abstract type AbstractPositionDependentRWMSampler <: AbstractMCMC.AbstractSampler end

struct HardPositionDependentRWMSampler{CF<:AbstractCovarianceField} <: AbstractPositionDependentRWMSampler
    CovF::CF
    Metadata::NGPCAJson.NGPCAUnitDO
end


# Initial step that will intialize state
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AbstractPositionDependentRWMSampler; kwargs)
    # TODO maybe have explicit state initialize function call here
    # dispatched on sampler type
    state = zeros(dimension(sampler))
    AbstractMCMC.step(rng, model, sampler, state; kwargs)

end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AbstractPositionDependentRWMSampler, state; kwargs)
    
    uniform_rv_draw = rand(rng)
    proposed_state = propose(rng, state, sampler)

    state
end

function covfield(sampler::HardPositionDependentRWMSampler)
    sampler.CovF
end

function metadata(sampler::HardPositionDependentRWMSampler)
    # Metadata should be some integer position indexable
    # sturcture that lines up with the position of the 
    # centroids inside the nearest neighbour lookup
    sampler.Metadata
end

function dimension(sampler::HardPositionDependentRWMSampler)
    metadata(sampler).m[begin]
end

function dimension(s::AbstractPositionDependentRWMSampler)
    error("Not Implemented for $(typeof(s))")
end

function propose(rng::AbstractRNG, state::AbstractVector, sampler::HardPositionDependentRWMSampler)
    idx, _ = get_knn(state, covfield(sampler), 1)
    covariance = construct_covariance(idx[begin], metadata(sampler))
    rand(rng, MvNormal(state, covariance))
end

function propose(rng::AbstractRNG, state::AbstractVector, sampler::AbstractPositionDependentRWMSampler)
    error("Not implemented for $(typeof(sampler))")
end

function construct_covariance(i::Int, indexableMetadata::NGPCAJson.NGPCAUnitDO)
    # TODO probably hide this under one more level which just takes index and stucture
    # that can be indexed into... But when I get around to abstracting the metdata
    eigvals, eigvecs = indexableMetadata.eigenvalues[:, i], indexableMetadata.weights[i]

    # TODO probably do this better... right now assume full dimensions so can recover
    # full covariance by Eigenvalue decomposition theorem
    # Also is there some optimization here at this multiplication?
    # Probably use a buffer since dimensionality is upper bounded
    eigvecs * Diagonal(eigvals) * transpose(eigvecs)

    # TODO Do I need to add ϵI term
    
end
