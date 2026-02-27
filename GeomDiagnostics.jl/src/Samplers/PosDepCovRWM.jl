
using AbstractMCMC
using ..NGPCAJson
using ..CovarianceFields
using Distributions, Random

abstract type AbstractPositionDependentRWMSampler <: AbstractMCMC.AbstractSampler end

struct HardPositionDependentRWMSampler{CF<:AbstractCovarianceField} <: AbstractPositionDependentRWMSampler
    CovF::CF
    Metadata::NGPCAJson.NGPCAUnitDO
end


# Initial step that will intialize state
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AbstractPositionDependentRWMSampler; kwargs)
    # TODO Need some way to get dimension nicely.
    # Probably imlement a dimension function + multiple dispatch
    state = zeros(sampler.CovF.Metadata.m[begin])
    AbstractMCMC.step(rng, model, sampler, state; kwargs)

end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AbstractPositionDependentRWMSampler, state; kwargs)
    
    uniform_rv_draw = rand(rng, 1)
    proposed_state = propose(rng, state, sampler)

    state
end

function propose(rng::AbstractRNG, state::AbstractVector, sampler::HardPositionDependentRWMSampler)
    idx, _ = get_knn(state, sampler.CovF, 1)
    covariance = construct_covariance(idx[begin], sampler.Metadata)
    rand(rng, MvNormal(state, covariance))

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
    
end
