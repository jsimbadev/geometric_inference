
using AbstractMCMC
using ..NGPCAJson
using ..CovarianceFields
using Distribution, Random

abstract type AbstractPositionDepenedentRWMSampler <: AbstractMCMC.AbstractSampler end

struct HardPositionDependentRWMSampler{CF<:AbstractCovarianceField} <: AbstractPositionDepenedentRWMSampler
    CovF::CF
    Metadata::NGPCAJson.NGPCAUnitDO
end


# Initial step that will intialize state
function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AbstractPositionDepenedentRWMSampler; kwargs)

    # Initialize state
    state = Vector([0.0, 0.0])
    AbstractMCMC.step(rng, model, sampler, state; kwargs)

end

function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, sampler::AbstractPositionDepenedentRWMSampler, state; kwargs)
    
    uniform_rv_draw = rand(rng, 1)
    proposed_state = propose(rng, state, sampler)

    state
end

function propose(rng::AbstractRNG, state::AbstractVector, sampler::HardPositionDependentRWMSampler)
    idx, _ = get_knn(state, sampler.CF, 1)
    covariance = construct_covariance(idx, sampler.Metadata)
    rand(rng, MvnNormal(state, covariance))

end

function construct_covariance(i::Int, indexableMetadata::NGPCAJson.NGPCAUnitDO)
    # TODO probably hide this under one more level which just takes index and stucture
    # that can be indexed into... But when I get around to abstracting the metdata
    eigvals, eigvecs = indexableMetadata.eigenvalues[:, i], indexableMetadata.weights[i]

    # TODO probably do this better... right now assume full dimensions so can recover
    # full covariance by Eigenvalue decomposition theorem
    # Also is there some optimization here at this multiplication?
    # Probably use a buffer since dimensionality is upper bounded
    transpose(eigvecs) * eigvals * eigvecs
    
end
